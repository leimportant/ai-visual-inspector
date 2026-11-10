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

This launches a Tkinter window where you can open an image, adjust parameters, and preview the transformation results live.

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
