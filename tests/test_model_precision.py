"""CPU correctness checks for the opt-in CUDA/CPU CNN BF16 policy.

These validate precision boundaries and gradients, not accelerator throughput.
"""

from dataclasses import asdict, fields

import pytest
import torch

from photograder.models import ModelConfig, build_model
from photograder.renderer import render


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _config(architecture="local", **kwargs):
    return ModelConfig(
        architecture=architecture, preview_size=32, map_size=8, lut_size=5, styles=("C", "A"), **kwargs
    )


def _assert_fp32_recipe(recipe):
    for field in fields(recipe):
        value = getattr(recipe, field.name)
        if isinstance(value, torch.Tensor):
            assert value.dtype == torch.float32, field.name
    for stage in recipe.stages or []:
        _assert_fp32_recipe(stage)


@pytest.mark.parametrize("architecture", ["fixed", "global", "local", "cascade", "unet", "bilateral"])
def test_bf16_neutral_initialization_preserves_fp32_render_and_gradient(architecture):
    torch.manual_seed(12)
    model = build_model(_config(architecture)).set_cnn_autocast("bf16")
    image = torch.rand(2, 3, 24, 32) * 0.7 + 0.1
    recipe = model.analyze(image, torch.tensor([0, 1]))
    _assert_fp32_recipe(recipe)
    output = render(image, recipe)
    assert output.dtype == torch.float32
    torch.testing.assert_close(output, image, atol=2e-6, rtol=1e-5)
    (output - image * 0.85).square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(g.dtype == torch.float32 and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_autocast_is_limited_to_convolutions_and_not_lut_mixing():
    model = build_model(_config(global_context=True)).set_cnn_autocast("bf16")
    seen = {}
    handles = []
    for name, module in [
        ("encoder", model.encoder.blocks[0][0]),
        ("decoder", model.decoder.output),
        ("mixer", model.mixer),
        ("bank", model.bank),
    ]:

        def record(_module, _inputs, output, label=name):
            seen[label] = output.dtype

        handles.append(module.register_forward_hook(record))
    # Even an accidental broad caller scope must not round learned LUT tables.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        recipe = model.analyze(torch.rand(1, 3, 24, 32))
    for handle in handles:
        handle.remove()
    assert seen == {
        "encoder": torch.bfloat16,
        "decoder": torch.bfloat16,
        "mixer": torch.float32,
        "bank": torch.float32,
    }
    _assert_fp32_recipe(recipe)
    assert all(p.dtype == torch.float32 for p in model.parameters())


def test_nonzero_bf16_local_model_is_close_to_fp32_and_has_live_gradients():
    torch.manual_seed(71)
    model = build_model(_config(global_context=True, basis_init="diverse", basis_init_strength=0.2))
    with torch.no_grad():
        model.decoder.output.weight.normal_(std=0.002)
        model.decoder.output.bias.copy_(torch.tensor([0.03, -0.02]))
        model.mixer.weight.normal_(std=0.02)
    image = torch.rand(2, 3, 24, 32) * 0.6 + 0.2
    styles = torch.tensor([0, 1])
    expected = model(image, styles).detach()
    model.set_cnn_autocast("bf16")
    actual = model(image, styles)
    # Quantized CNN activations can differ; rendering itself remains FP32.
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=2e-3)
    (actual - image * 0.85).square().mean().backward()
    for parameter in (
        model.encoder.blocks[0][0].weight,
        model.decoder.output.weight,
        model.mixer.weight,
        model.bank.tables,
    ):
        assert parameter.grad is not None
        assert parameter.grad.dtype == torch.float32
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_bf16_repair_has_no_global_gradients():
    model = build_model(_config()).set_cnn_autocast("bf16")
    image = torch.rand(1, 3, 24, 32) * 0.7 + 0.1
    recipe = model.analyze(image, bypass_global=True)
    _assert_fp32_recipe(recipe)
    assert recipe.luts is None
    (render(image, recipe, grade_strength=0) - image * 0.8).square().mean().backward()
    assert model.bank.tables.grad is None
    assert model.mixer.weight.grad is None
    assert torch.isfinite(model.decoder.output.weight.grad).all()
    assert model.decoder.output.weight.grad.abs().sum() > 0


def test_autocast_policy_has_no_checkpoint_keys_and_can_be_disabled_exactly():
    model = build_model(_config("cascade"))
    image = torch.rand(1, 3, 24, 32) * 0.7 + 0.1
    config_before = asdict(model.config)
    keys_before = set(model.state_dict())
    expected = model(image).detach()
    model.set_cnn_autocast("bf16")
    assert model.exposure_model.cnn_autocast == "bf16"
    assert model.contrast_model.cnn_autocast == "bf16"
    assert model.global_model.cnn_autocast == "bf16"
    assert set(model.state_dict()) == keys_before
    assert asdict(model.config) == config_before
    restored = build_model(_config("cascade"))
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.cnn_autocast == "none"
    assert restored.exposure_model.cnn_autocast == "none"
    model.set_cnn_autocast("none")
    torch.testing.assert_close(model(image), expected, atol=0, rtol=0)
    torch.testing.assert_close(restored(image), expected, atol=0, rtol=0)


def test_precision_policy_rejects_unknown_mode_and_converted_parameters():
    model = build_model(_config())
    with pytest.raises(ValueError, match="CNN autocast"):
        model.set_cnn_autocast("fp16")
    with pytest.raises(ValueError, match="parameters FP32"):
        model.bfloat16().set_cnn_autocast("bf16")


@pytest.mark.parametrize("style", [torch.tensor([-1, 0]), torch.tensor([0, 2]), -1, 2])
def test_cpu_style_bounds_still_raise_value_error(style):
    model = build_model(_config("fixed"))
    with pytest.raises(ValueError, match="style index outside"):
        model.analyze(torch.rand(2, 3, 16, 16), style)


def test_style_shape_and_both_valid_bounds():
    model = build_model(_config("fixed"))
    image = torch.rand(2, 3, 16, 16)
    recipe = model.analyze(image, torch.tensor([0, 1]))
    torch.testing.assert_close(recipe.luts, model.bank.tables, atol=0, rtol=0)
    with pytest.raises(ValueError, match="one index per image"):
        model.analyze(image, torch.tensor([[0], [1]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_valid_style_bounds_and_backward():
    # Do not issue an invalid device assertion here: CUDA correctly invalidates
    # the execution stream after one, which would poison unrelated tests.
    model = build_model(_config("fixed")).cuda()
    image = torch.rand(2, 3, 16, 16, device="cuda") * 0.7 + 0.1
    recipe = model.analyze(image, torch.tensor([0, 1], device="cuda"))
    output = render(image, recipe)
    torch.testing.assert_close(output, image, atol=2e-6, rtol=1e-5)
    output.mean().backward()
    assert torch.isfinite(model.bank.tables.grad).all()
