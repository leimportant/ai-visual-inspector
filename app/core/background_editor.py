import sys
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image


def _fit_background(background: Image.Image, size: Tuple[int, int]) -> Image.Image:
    width, height = size
    scale = max(width / background.width, height / background.height)
    resized = background.resize(
        (int(background.width * scale), int(background.height * scale)),
        getattr(Image, "Resampling", Image).LANCZOS,
    )
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def replace_background(
    image_path: str,
    output_path: str,
    background_color: Optional[Tuple[int, int, int]] = None,
    background_image_path: Optional[str] = None,
) -> Path:
    if (background_color is None) == (background_image_path is None):
        raise ValueError("Pilih tepat satu jenis background: warna atau gambar.")

    if background_color is not None and (
        len(background_color) != 3
        or any(not isinstance(channel, int) or not 0 <= channel <= 255 for channel in background_color)
    ):
        raise ValueError("Warna background harus berupa tiga nilai RGB antara 0 dan 255.")

    try:
        from rembg import remove
    except ModuleNotFoundError as exc:
        if exc.name == "rembg":
            raise RuntimeError(
                "rembg tidak tersedia untuk Python yang menjalankan aplikasi "
                f"({sys.executable}). Jalankan aplikasi dengan interpreter tempat "
                "rembg terpasang, misalnya './.venv/bin/python app/main.py --ui', "
                "atau install dengan interpreter aktif: "
                f"'{sys.executable} -m pip install -r requirements.txt'."
            ) from exc
        raise RuntimeError(
            f"rembg gagal dimuat karena modul dependensi '{exc.name}' tidak tersedia "
            f"untuk Python {sys.executable}: {exc}"
        ) from exc
    except ImportError as exc:
        raise RuntimeError(
            f"rembg gagal dimuat oleh Python {sys.executable}: {exc}"
        ) from exc

    source_path = Path(image_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Gambar sumber tidak ditemukan: {image_path}")

    with Image.open(source_path) as source_file:
        foreground = remove(source_file.convert("RGBA")).convert("RGBA")

    if background_color is not None:
        background = Image.new("RGBA", foreground.size, background_color + (255,))
    else:
        background_path = Path(background_image_path)
        if not background_path.is_file():
            raise FileNotFoundError(
                f"Gambar background tidak ditemukan: {background_image_path}"
            )
        with Image.open(background_path) as background_file:
            background = _fit_background(
                background_file.convert("RGBA"),
                foreground.size,
            )

    result = Image.alpha_composite(background, foreground).convert("RGB")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.save(destination, format="PNG")
    return destination
