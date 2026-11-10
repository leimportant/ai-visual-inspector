
import numpy as np
import pytest

from app.core.image_processor import (
    to_grayscale,
    resize_image,
    apply_gaussian_blur,
    apply_threshold,
    detect_contours,
    draw_contours,
    detect_edges,
)


@pytest.fixture
def sample_image():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_grayscale(sample_image):
    result = to_grayscale(sample_image)

    assert result.shape == (480, 640)


def test_resize(sample_image):
    result = resize_image(sample_image, width=320)

    assert result.shape == (240, 320, 3)


def test_gaussian_blur(sample_image):
    result = apply_gaussian_blur(sample_image)

    assert result.shape == sample_image.shape


def test_invalid_kernel(sample_image):
    with pytest.raises(ValueError):
        apply_gaussian_blur(sample_image, kernel_size=4)


def test_thresholding(sample_image):
    result = apply_threshold(sample_image, threshold_value=127)

    assert result.shape == sample_image.shape[:2]
    assert result.dtype == np.uint8


def test_contour_detection(sample_image):
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255

    contours = detect_contours(mask, threshold_value=127)
    result = draw_contours(mask, contours)

    assert len(contours) > 0
    assert result.shape == (100, 100, 3)


def test_edge_detection(sample_image):
    result = detect_edges(sample_image)

    assert result.shape == (480, 640)
    assert result.dtype == np.uint8
