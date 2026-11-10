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
    assert generator.generate("buatkan gambar kelinci") == "generated image"
    assert calls == [
        ("load", "test-model", "float32"),
        ("device", "cpu"),
        (
            "generate",
            "a flower",
            {
                "num_inference_steps": 35,
                "guidance_scale": 7.5,
                "negative_prompt": (
                    "blurry, low quality, distorted, extra limbs, duplicate subjects, "
                    "text, watermark"
                ),
            },
        ),
        (
            "generate",
            "Photorealistic wildlife photo of a real rabbit, a small animal with long "
            "ears, soft fur, and four legs",
            {
                "num_inference_steps": 35,
                "guidance_scale": 7.5,
                "negative_prompt": (
                    "blurry, low quality, distorted, extra limbs, duplicate subjects, "
                    "text, watermark"
                ),
            },
        ),
    ]


def test_prepare_prompt_distinguishes_automatic_scooter_from_cruiser():
    prompt, negative_prompt = ImageGenerator._prepare_prompt(
        "Buatkan gambar motor matic berwarna merah"
    )

    assert prompt == (
        "Photorealistic product photo of automatic step-through scooter with a flat "
        "floorboard and compact wheels berwarna merah"
    )
    assert "Harley-Davidson" in negative_prompt
    assert "cruiser motorcycle" in negative_prompt


def test_prepare_prompt_preserves_english_prompt():
    prompt, negative_prompt = ImageGenerator._prepare_prompt("a red flower")

    assert prompt == "a red flower"
    assert "blurry" in negative_prompt
