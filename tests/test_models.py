import pytest
import torch

from photograder.models import ModelConfig, build_model, initialize_from_model
from photograder.renderer import render


@pytest.fixture(autouse=True)
def reduce_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("architecture", ["fixed", "global", "local", "cascade", "unet", "bilateral"])
def test_all_architectures_neutral_initialization_and_backward(architecture):
    model = build_model(
        ModelConfig(architecture=architecture, preview_size=64, map_size=16, lut_size=9, styles=("A", "B"))
    )
    image = torch.rand(2, 3, 39, 61) * 0.8 + 0.1
    recipe = model.analyze(image, torch.tensor([0, 1]))
    output = render(image, recipe)
    torch.testing.assert_close(output, image, atol=2e-6, rtol=1e-5)
    (output - image * 0.9).square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert any(grad.abs().sum() > 0 for grad in grads)


def test_shared_model_parameter_budget_and_aspect_ratio():
    model = build_model()
    assert sum(p.numel() for p in model.parameters()) < 2_000_000
    recipe = model.analyze(torch.rand(1, 3, 100, 200))
    assert recipe.exposure.shape == (1, 1, 32, 64)
    assert recipe.guide.shape == recipe.exposure.shape
    assert recipe.source_size == (100, 200)


def test_lut_mixer_gets_gradient_from_initial_identity_average():
    model = build_model(ModelConfig(architecture="global", preview_size=64, lut_size=9))
    image = torch.rand(1, 3, 33, 47) * 0.5 + 0.2
    (model(image) - image * 0.8).square().mean().backward()
    assert model.mixer.weight.grad.abs().sum() > 0


def test_repair_bypass_has_no_hidden_global_transform_or_gradient():
    model = build_model(ModelConfig(preview_size=64, map_size=16, lut_size=9))
    image = torch.rand(1, 3, 32, 48) * 0.7 + 0.1
    with torch.no_grad():
        model.bank.tables.mul_(0.5)
    recipe = model.analyze(image, bypass_global=True)
    assert recipe.luts is None
    torch.testing.assert_close(render(image, recipe, grade_strength=0), image, atol=2e-6, rtol=1e-5)
    (render(image, recipe, grade_strength=0) - image * 0.9).square().mean().backward()
    assert model.bank.tables.grad is None
    assert model.mixer.weight.grad is None
    assert model.decoder.output.weight.grad.abs().sum() > 0


def test_tonal_specialists_start_neutral_and_can_learn_both_directions():
    model = build_model(ModelConfig(preview_size=64, map_size=16, lut_size=9, tonal_specialists=True))
    image = torch.rand(1, 3, 32, 48)
    recipe = model.analyze(image, bypass_global=True)
    assert torch.count_nonzero(recipe.exposure) == 0
    (render(image, recipe) - image * 0.9).square().mean().backward()
    assert model.decoder.output.weight.grad[0].abs().sum() > 0
    assert model.decoder.output.weight.grad[1].abs().sum() > 0


def test_style_bounds_fail_clearly():
    model = build_model(ModelConfig(architecture="fixed"))
    with pytest.raises(ValueError, match="style index"):
        model.analyze(torch.rand(1, 3, 4, 4), torch.tensor([1]))


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_shared_model_mps_forward_backward_parity():
    torch.manual_seed(41)
    config = ModelConfig(preview_size=64, map_size=16, lut_size=9)
    cpu = build_model(config)
    # Nonzero spatial fields exercise the decoder and guided rendering paths.
    with torch.no_grad():
        cpu.decoder.output.weight.normal_(std=0.002)
        cpu.mixer.weight.normal_(std=0.005)
    mps = build_model(config).to("mps")
    mps.load_state_dict(cpu.state_dict())
    x = torch.rand(1, 3, 40, 53) * 0.6 + 0.2
    left = cpu(x)
    right = mps(x.to("mps"))
    torch.testing.assert_close(left, right.cpu(), atol=5e-5, rtol=5e-4)
    left.square().mean().backward()
    right.square().mean().backward()
    for cpu_param, mps_param in zip(cpu.parameters(), mps.parameters(), strict=False):
        if cpu_param.grad is not None:
            assert torch.isfinite(mps_param.grad).all()
            torch.testing.assert_close(cpu_param.grad, mps_param.grad.cpu(), atol=4e-5, rtol=2e-2)


@pytest.mark.parametrize("architecture", ["global", "local", "cascade", "unet", "bilateral"])
def test_initialization_matches_named_style_rows_and_keeps_new_styles_fresh(architecture):
    source = build_model(ModelConfig(architecture=architecture, styles=("A", "B"), lut_size=5))
    target = build_model(ModelConfig(architecture=architecture, styles=("C", "B", "A"), lut_size=5))
    fresh = {name: value.detach().clone() for name, value in target.state_dict().items()}
    with torch.no_grad():
        for name, value in source.named_parameters():
            if name.endswith("style_embedding.weight"):
                value[0].fill_(0.12)
                value[1].fill_(0.34)
    report = initialize_from_model(target, source)
    assert report["new_styles"] == ["C"]
    assert report["style_rows_copied"]
    for name, value in target.state_dict().items():
        if name.endswith("style_embedding.weight"):
            torch.testing.assert_close(value[0], fresh[name][0], atol=0, rtol=0)
            torch.testing.assert_close(value[1], torch.full_like(value[1], 0.34))
            torch.testing.assert_close(value[2], torch.full_like(value[2], 0.12))
        elif name in source.state_dict() and value.shape == source.state_dict()[name].shape:
            torch.testing.assert_close(value, source.state_dict()[name], atol=0, rtol=0)


def test_fixed_lut_initialization_remaps_style_tables():
    source = build_model(ModelConfig(architecture="fixed", styles=("A", "B"), lut_size=5))
    target = build_model(ModelConfig(architecture="fixed", styles=("B", "C", "A"), lut_size=5))
    fresh = target.bank.tables[1].detach().clone()
    with torch.no_grad():
        source.bank.tables[0].mul_(0.8)
        source.bank.tables[1].mul_(0.6)
    report = initialize_from_model(target, source)
    torch.testing.assert_close(target.bank.tables[0], source.bank.tables[1])
    torch.testing.assert_close(target.bank.tables[2], source.bank.tables[0])
    torch.testing.assert_close(target.bank.tables[1], fresh, atol=0, rtol=0)
    assert len(report["style_rows_copied"]["bank.tables"]) == 2


def test_fixed_style_tables_are_never_mistaken_for_global_lut_bases():
    source = build_model(ModelConfig(architecture="fixed", styles=("A", "B"), lut_size=5))
    target = build_model(ModelConfig(architecture="global", styles=("A", "B"), num_luts=2, lut_size=5))
    with pytest.raises(ValueError, match="No compatible weights"):
        initialize_from_model(target, source)


def test_context_statistics_ignore_letterbox_and_invalid_regions():
    from photograder.models import _preview

    model = build_model(ModelConfig(architecture="global", global_context=True, preview_size=32, lut_size=5))
    image = torch.rand(1, 3, 16, 32)
    preview, crop = _preview(image, 32)
    rectangular = model._context(preview, crop)
    mask = torch.zeros(1, 1, 32, 32)
    mask[..., 8:24, :] = 1
    square = model._context(preview, (0, 0, 32, 32), mask)
    torch.testing.assert_close(rectangular, square, atol=1e-6, rtol=1e-6)
    preview[..., :8, :] = 0.99
    preview[..., 24:, :] = 0.01
    torch.testing.assert_close(model._context(preview, (0, 0, 32, 32), mask), square, atol=0, rtol=0)


def test_diverse_context_variant_starts_identity_with_live_context_gradients():
    model = build_model(
        ModelConfig(
            architecture="global",
            global_context=True,
            basis_init="diverse",
            basis_init_strength=0.2,
            preview_size=32,
            lut_size=5,
        )
    )
    image = torch.rand(2, 3, 24, 32) * 0.6 + 0.2
    output = model(image)
    torch.testing.assert_close(output, image, atol=3e-7, rtol=1e-5)
    (output - image * 0.8).square().mean().backward()
    assert model.mixer.weight.shape[-1] == 152
    assert model.mixer.weight.grad[:, -8:].abs().sum() > 0
    centered = model.bank.tables.detach().flatten(1)
    centered = centered - centered.mean(0, keepdim=True)
    assert torch.linalg.matrix_rank(centered, atol=1e-5) == 7


def test_old_configuration_keeps_legacy_architecture_state_compatible():
    old_config = {
        "architecture": "global",
        "styles": ["Expert C"],
        "preview_size": 64,
        "lut_size": 5,
        "num_luts": 8,
        "map_size": 16,
        "upsampling": "guided",
        "order": "global_first",
        "tonal_specialists": False,
    }
    model = build_model(ModelConfig(**old_config))
    assert model.mixer.weight.shape[-1] == 144
    replica = build_model(ModelConfig(**old_config))
    replica.load_state_dict(model.state_dict(), strict=True)
    assert not replica.config.global_context
