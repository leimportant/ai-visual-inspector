import sys
from types import SimpleNamespace

import pytest

from app.core.image_generator import ImageGenerator


def test_generate_rejects_empty_prompt():
    generator = ImageGenerator()

    with pytest.raises(ValueError, match="tidak boleh kosong"):
        generator.generate("  ")


def test_generate_loads_pipeline_once(monkeypatch):
    calls = []

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_id, torch_dtype):
            calls.append(("load", model_id, torch_dtype))
            return cls()

        def to(self, device):
            calls.append(("device", device))
            return self

        def __call__(self, prompt, **kwargs):
            calls.append(("generate", prompt, kwargs))
            return SimpleNamespace(images=["generated image"])

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        float16="float16",
        float32="float32",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(StableDiffusionPipeline=FakePipeline),
    )

    generator = ImageGenerator(model_id="test-model")

    assert generator.generate("a flower") == "generated image"
    assert generator.generate("a motorcycle") == "generated image"
    assert calls == [
        ("load", "test-model", "float32"),
        ("device", "cpu"),
        ("generate", "a flower", {"num_inference_steps": 25, "guidance_scale": 7.5}),
        ("generate", "a motorcycle", {"num_inference_steps": 25, "guidance_scale": 7.5}),
    ]
