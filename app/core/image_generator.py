import os
import re
from typing import Optional


DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"
_NEGATIVE_PROMPT = "blurry, low quality, distorted, extra limbs, duplicate subjects, text, watermark"


class ImageGenerator:
    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or os.getenv("AI_VISUAL_INSPECTOR_MODEL", DEFAULT_MODEL_ID)
        self._pipeline = None

    def generate(self, prompt: str):
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Deskripsi gambar tidak boleh kosong.")

        if self._pipeline is None:
            self._load_pipeline()

        prompt, negative_prompt = self._prepare_prompt(prompt)
        return self._pipeline(
            prompt,
            num_inference_steps=35,
            guidance_scale=7.5,
            negative_prompt=negative_prompt,
        ).images[0]

    @staticmethod
    def _prepare_prompt(prompt: str):
        prompt = re.sub(
            r"^(?:(?:tolong|please)\s+)?(?:buatkan|buat)\s+"
            r"(?:(?:gambar|foto|ilustrasi)\s+)?",
            "",
            prompt.strip(),
            flags=re.IGNORECASE,
        )
        prompt = re.sub(r"^(?:gambar|foto|ilustrasi)\s+", "", prompt, flags=re.IGNORECASE)

        if re.search(r"\b(?:motor|skuter)\s+matic\b", prompt, flags=re.IGNORECASE):
            prompt = re.sub(
                r"\b(?:motor|skuter)\s+matic\b",
                "automatic step-through scooter with a flat floorboard and compact wheels",
                prompt,
                flags=re.IGNORECASE,
            )
            prompt = f"Photorealistic product photo of {prompt}"
            negative_prompt = (
                f"{_NEGATIVE_PROMPT}, Harley-Davidson, cruiser motorcycle, chopper"
            )
        elif re.search(r"\bkelinci\b", prompt, flags=re.IGNORECASE):
            prompt = re.sub(
                r"\b(?:seekor\s+)?kelinci\b",
                "a real rabbit, a small animal with long ears, soft fur, and four legs",
                prompt,
                flags=re.IGNORECASE,
            )
            prompt = f"Photorealistic wildlife photo of {prompt}"
            negative_prompt = _NEGATIVE_PROMPT
        else:
            negative_prompt = _NEGATIVE_PROMPT

        return prompt, negative_prompt

    def _load_pipeline(self):
        try:
            import torch
            from diffusers import StableDiffusionPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Dependensi pembuat gambar belum tersedia. Jalankan "
                "'python3 -m pip install -r requirements.txt'."
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
        ).to(device)
