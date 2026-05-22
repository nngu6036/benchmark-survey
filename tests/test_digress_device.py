from types import SimpleNamespace

import torch

from empirical_comparison.models.wrappers.digress import (
    DiGressWrapper,
    _cuda_device_is_usable,
    _patch_torchvision_onnx_compat,
)
from empirical_comparison.utils.torch_compat import torch_load_compat


def test_cuda_probe_rejects_unusable_cuda(monkeypatch):
    def raise_cuda_init(*args, **kwargs):
        if torch.device(kwargs.get("device", "cpu")).type == "cuda":
            raise RuntimeError("The NVIDIA driver on your system is too old")
        return original_empty(*args, **kwargs)

    original_empty = torch.empty
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "empty", raise_cuda_init)

    usable, reason = _cuda_device_is_usable("cuda")

    assert not usable
    assert "NVIDIA driver" in str(reason)


def test_digress_device_falls_back_to_cpu_when_cuda_probe_fails(monkeypatch):
    def raise_cuda_init(*args, **kwargs):
        if torch.device(kwargs.get("device", "cpu")).type == "cuda":
            raise RuntimeError("The NVIDIA driver on your system is too old")
        return original_empty(*args, **kwargs)

    original_empty = torch.empty
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "empty", raise_cuda_init)

    wrapper = object.__new__(DiGressWrapper)

    assert wrapper._resolve_device(None) == "cpu"
    assert wrapper._resolve_device("cuda") == "cpu"


def test_torchvision_onnx_cast_long_compat_patch(monkeypatch):
    from torch.onnx import symbolic_opset9

    monkeypatch.delattr(symbolic_opset9, "_cast_Long", raising=False)

    _patch_torchvision_onnx_compat()

    graph = SimpleNamespace(op=lambda *args, **kwargs: (args, kwargs))
    args, kwargs = symbolic_opset9._cast_Long(graph, "value")
    assert args == ("Cast", "value")
    assert kwargs == {"to_i": 7}


def test_legacy_torch_load_retries_without_weights_only_for_old_torch(monkeypatch):
    calls = []

    def old_torch_load(*args, **kwargs):
        calls.append(kwargs.copy())
        if "weights_only" in kwargs:
            raise TypeError("_Unpickler.__init__() got an unexpected keyword argument 'weights_only'")
        return "loaded"

    monkeypatch.setattr(torch, "load", old_torch_load)
    wrapper = object.__new__(DiGressWrapper)

    with wrapper._legacy_torch_load():
        assert torch.load("dataset.pt") == "loaded"

    assert calls == [{"weights_only": False}, {}]


def test_torch_load_compat_retries_without_weights_only_for_old_torch(monkeypatch):
    calls = []

    def old_torch_load(*args, **kwargs):
        calls.append(kwargs.copy())
        if "weights_only" in kwargs:
            raise TypeError("'weights_only' is an invalid keyword argument for Unpickler()")
        return {"checkpoint": True}

    monkeypatch.setattr(torch, "load", old_torch_load)

    assert torch_load_compat("model.pt", map_location="cpu", weights_only=False) == {"checkpoint": True}
    assert calls == [{"map_location": "cpu", "weights_only": False}, {"map_location": "cpu"}]
