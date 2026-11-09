
from pathlib import Path

import cv2

from app.utils.image_loader import load_image
from app.core.image_processor import (
    to_grayscale,
    resize_image,
    apply_gaussian_blur,
    detect_edges,
)


INPUT_DIR = Path("data/input")
OUTPUT_DIR = Path("data/output")


def main():
    image_files = list(INPUT_DIR.glob("*"))

    if not image_files:
        print("No images found in data/input/")
        return

    image_path = image_files[0]

    image = load_image(str(image_path))

    resized = resize_image(image)
    grayscale = to_grayscale(resized)
    blurred = apply_gaussian_blur(resized)
    edges = detect_edges(resized)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "original.jpg"),
        resized
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "grayscale.jpg"),
        grayscale
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "blurred.jpg"),
        blurred
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "edges.jpg"),
        edges
    )

    print("Image processing completed!")
    print(f"Input: {image_path}")
    print(f"Output: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
