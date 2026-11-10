import os
from typing import Optional


DEFAULT_MODEL_ID = "segmind/tiny-sd"


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

        return self._pipeline(
            prompt,
            num_inference_steps=25,
            guidance_scale=7.5,
        ).images[0]

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
