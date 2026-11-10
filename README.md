# AI Visual Inspector

## Overview
AI Visual Inspector is a lightweight Python project for basic image processing and computer vision tasks using OpenCV. It reads an image from the `data/input` directory, applies several preprocessing operations, and stores the results in `data/output`.

This project is intended for learning and experimentation with common image processing workflows such as grayscale conversion, resizing, blur, and edge detection.

## Features
- Load an image from disk
- Convert the image to grayscale
- Resize the image to a target width
- Apply Gaussian blur for smoothing
- Detect image edges with Canny edge detection
- Apply binary thresholding to isolate intensity regions
- Extract and visualize contours from thresholded images
- Launch a simple Tkinter preview interface for live inspection of transformations
- Generate new images from text descriptions with a locally run diffusion model
- Replace an image background with a solid color or another image
- Save processed outputs to the output directory
- Unit tests for core image processing functions

## Technology Stack
- Python 3.9+
- OpenCV (`opencv-python`)
- NumPy
- Pillow
- Matplotlib
- Pytest

## Project Structure
```text
.
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── image_processor.py
│   └── utils/
│       ├── __init__.py
│       └── image_loader.py
├── data/
│   ├── input/
│   └── output/
├── tests/
│   └── test_image_processor.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation
Clone the repository and install the dependencies:

```bash
git clone <repository-url>
cd ai-visual-inspector
python3 -m pip install -r requirements.txt
```

## Usage
Place the image you want to process inside the `data/input/` folder, then run:

```bash
python3 app/main.py
```

You can also run it as a module:

```bash
python3 -m app.main
```

To customize the processing behavior, you can pass command-line arguments:

```bash
python3 app/main.py --input-dir data/input --output-dir data/output --width 320 --kernel-size 7 --threshold1 80 --threshold2 160
```

You can process a single image or multiple images in one run:

```bash
python3 app/main.py data/input/alva-cervo.jpg data/input/another-image.jpg --width 480 --kernel-size 5
```

If no image file is specified, the program automatically processes all image files found in `data/input/`.

The script will process the selected image(s) and save the following files into the output directory:

- `original.jpg`
- `grayscale.jpg`
- `blurred.jpg`
- `edges.jpg`
- `thresholded.jpg`
- `contours.jpg`

To open a simple graphical preview instead of batch processing, run:

```bash
python3 app/main.py --ui
```

This launches a Tkinter window where you can open an image, adjust parameters, and preview the transformation results live. The **Buat Gambar** tab accepts a text description and saves generated images to `data/generated/`.

The **Ganti Background** tab can replace the background of an opened image or generated image with a chosen color or another image. Results are saved as PNG files in `data/backgrounds/`. Background removal runs locally; its model is downloaded on first use and cached for later runs.

Image generation runs locally with `runwayml/stable-diffusion-v1-5` by default, replacing the smaller `segmind/tiny-sd` model for better prompt adherence. The model is downloaded the first time it is used, so an internet connection and several gigabytes of free disk space are needed; subsequent generation uses the cached model. Generation can be slow on CPU. Indonesian requests for rabbits and automatic scooters are expanded into more specific English prompts, and scooter prompts explicitly discourage cruiser and chopper designs. To use a different Diffusers-compatible model or a model directory already on disk, set `AI_VISUAL_INSPECTOR_MODEL` before starting the application.

The generation and background-removal dependencies (`torch`, `diffusers`, `transformers`, `accelerate`, and `rembg`) are included in `requirements.txt`. For GPU acceleration, install the PyTorch build that matches your CUDA setup by following the [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

Run the app with the same Python environment used to install its dependencies. For example, on Linux or macOS with a project virtual environment, use `./.venv/bin/python app/main.py --ui` (and install dependencies with `./.venv/bin/python -m pip install -r requirements.txt`).

## Image Processing Pipeline
The processing flow implemented by the project is:

1. Load the image from disk
2. Resize the image to a standard size
3. Convert to grayscale
4. Apply Gaussian blur
5. Run edge detection using Canny
6. Apply binary thresholding to segment the foreground/background
7. Extract and draw contours from the thresholded image
8. Save each output to the output directory

This pipeline is a simple introduction to image enhancement and feature extraction techniques.

## Testing
Run the test suite using:

```bash
pytest -q
```

The tests cover:
- grayscale conversion
- resize behavior
- Gaussian blur application
- invalid kernel validation
- thresholding behavior
- contour detection and contour drawing
- edge detection output shape and dtype

## Code Coverage
To measure coverage, install `pytest-cov` and run:

```bash
pip install pytest-cov
pytest --cov=app --cov-report=term-missing
```

This helps evaluate how much of the project is exercised by the automated tests.

## Output Examples
The project generates processed images such as:

- original image
- grayscale version
- blurred version
- edge-highlighted output
- thresholded binary mask
- contour-highlighted image

These outputs can be used as quick visual comparisons for understanding how each image transformation changes the source data.

## Future Improvements
Possible improvements for this project include:
- adding support for multiple image files in one run
- supporting batch processing and export formats beyond JPEG
