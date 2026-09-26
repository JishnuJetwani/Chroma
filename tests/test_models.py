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


@pytest.mark.parametrize("architecture", ["fixed", "global"])
def test_all_architectures_neutral_initialization_and_backward(architecture):
    model = build_model(
        ModelConfig(architecture=architecture, preview_size=64, lut_size=9, styles=("A", "B"))
    )
    image = torch.rand(2, 3, 39, 61) * 0.8 + 0.1
    recipe = model.analyze(image, torch.tensor([0, 1]))
    output = render(image, recipe)
    torch.testing.assert_close(output, image, atol=2e-06, rtol=1e-05)
    (output - image * 0.9).square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert any(grad.abs().sum() > 0 for grad in grads)


def test_lut_mixer_gets_gradient_from_initial_identity_average():
    model = build_model(ModelConfig(architecture="global", preview_size=64, lut_size=9))
    image = torch.rand(1, 3, 33, 47) * 0.5 + 0.2
    (model(image) - image * 0.8).square().mean().backward()
    assert model.mixer.weight.grad.abs().sum() > 0


def test_style_bounds_fail_clearly():
    model = build_model(ModelConfig(architecture="fixed"))
    with pytest.raises(ValueError, match="style index"):
        model.analyze(torch.rand(1, 3, 4, 4), torch.tensor([1]))


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
