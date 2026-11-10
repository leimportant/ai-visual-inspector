
import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import cv2

from app.utils.image_loader import load_image
from app.core.image_processor import (
    to_grayscale,
    resize_image,
    apply_gaussian_blur,
    apply_threshold,
    detect_contours,
    draw_contours,
    detect_edges,
)

try:
    from app.ui import main as launch_ui
except (ModuleNotFoundError, RuntimeError):
    def launch_ui():
        raise RuntimeError(
            "Tkinter is not installed. Install python3-tk to use the --ui preview mode."
        )


DEFAULT_INPUT_DIR = Path("data/input")
DEFAULT_OUTPUT_DIR = Path("data/output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process an image with OpenCV filters and save the results."
    )
    parser.add_argument(
        "image_paths",
        nargs="*",
        help="Optional image files to process. If omitted, all images in --input-dir are processed.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing input images. Default: data/input",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where processed images will be saved. Default: data/output",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Target width for resizing the image. Default: 640",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=5,
        help="Gaussian blur kernel size. Must be a positive odd integer. Default: 5",
    )
    parser.add_argument(
        "--threshold1",
        type=int,
        default=100,
        help="Lower Canny threshold. Default: 100",
    )
    parser.add_argument(
        "--threshold2",
        type=int,
        default=200,
        help="Upper Canny threshold. Default: 200",
    )
    parser.add_argument(
        "--threshold-value",
        type=int,
        default=127,
        help="Binary threshold value for thresholding and contour extraction. Default: 127",
    )
    parser.add_argument(
        "--max-value",
        type=int,
        default=255,
        help="Maximum intensity used for binary thresholding. Default: 255",
    )
    parser.add_argument(
        "--contour-thickness",
        type=int,
        default=2,
        help="Thickness of contour lines drawn in the output image. Default: 2",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch a simple Tkinter preview window instead of batch processing images.",
    )
    return parser


def resolve_image_paths(image_paths: Optional[Sequence[str]], input_dir: Path) -> list[Path]:
    if image_paths:
        return [Path(path) for path in image_paths]

    image_files = sorted(path for path in input_dir.glob("*") if path.is_file())

    if not image_files:
        raise FileNotFoundError(f"No images found in {input_dir}/")

    return image_files


def process_image(
    image_path: Path,
    output_dir: Path,
    width: int = 640,
    kernel_size: int = 5,
    threshold1: int = 100,
    threshold2: int = 200,
    threshold_value: int = 127,
    max_value: int = 255,
    contour_thickness: int = 2,
) -> Path:
    image = load_image(str(image_path))

    resized = resize_image(image, width=width)
    grayscale = to_grayscale(resized)
    blurred = apply_gaussian_blur(resized, kernel_size=kernel_size)
    edges = detect_edges(resized, threshold1=threshold1, threshold2=threshold2)
    thresholded = apply_threshold(resized, threshold_value=threshold_value, max_value=max_value)
    contours = detect_contours(resized, threshold_value=threshold_value, max_value=max_value)
    contour_image = draw_contours(
        resized,
        contours,
        thickness=contour_thickness,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_dir / "original.jpg"), resized)
    cv2.imwrite(str(output_dir / "grayscale.jpg"), grayscale)
    cv2.imwrite(str(output_dir / "blurred.jpg"), blurred)
    cv2.imwrite(str(output_dir / "edges.jpg"), edges)
    cv2.imwrite(str(output_dir / "thresholded.jpg"), thresholded)
    cv2.imwrite(str(output_dir / "contours.jpg"), contour_image)

    return output_dir


def main(argv: Optional[Sequence[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.ui:
        launch_ui()
        return 0

    try:
        image_paths = resolve_image_paths(args.image_paths, args.input_dir)
        output_dir = args.output_dir

        for image_path in image_paths:
            process_image(
                image_path=image_path,
                output_dir=output_dir,
                width=args.width,
                kernel_size=args.kernel_size,
                threshold1=args.threshold1,
                threshold2=args.threshold2,
                threshold_value=args.threshold_value,
                max_value=args.max_value,
                contour_thickness=args.contour_thickness,
            )
            print("Image processing completed!")
            print(f"Input: {image_path}")
            print(f"Output: {output_dir.resolve()}")
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
