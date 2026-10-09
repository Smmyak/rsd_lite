import importlib.util

import pytest

torch_spec = importlib.util.find_spec("torch")
pytestmark = pytest.mark.skipif(torch_spec is None, reason="torch is not installed")

if torch_spec is not None:
    import torch
    from rsdehamba_lite.model import RSDehambaLite, count_parameters


def test_shape_preserved():
    model = RSDehambaLite(base_dim=8, depths=[1, 1, 1])
    x = torch.rand(2, 3, 32, 48)
    y = model(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_gradients_flow():
    model = RSDehambaLite(base_dim=8, depths=[1, 1, 1])
    x = torch.rand(1, 3, 32, 32)
    y = model(x).mean()
    y.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_lite_parameter_budget_under_paper_reported_baseline():
    model = RSDehambaLite(base_dim=32, depths=[1, 2, 2])
    assert count_parameters(model) < 1_800_000

