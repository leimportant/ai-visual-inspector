import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from app.core.background_editor import replace_background


def test_missing_rembg_reports_active_interpreter(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGB", (2, 2)).save(source)
    monkeypatch.setitem(sys.modules, "rembg", None)

    with pytest.raises(RuntimeError) as exc:
        replace_background(
            str(source),
            str(tmp_path / "result.png"),
            background_color=(255, 255, 255),
        )

    assert "interpreter tempat rembg terpasang" in str(exc.value)
    assert "./.venv/bin/python app/main.py --ui" in str(exc.value)


def test_replace_background_with_color(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    output = tmp_path / "result.png"
    Image.new("RGB", (2, 2), (0, 0, 0)).save(source)
    foreground = Image.new("RGBA", (2, 2), (255, 0, 0, 0))
    foreground.putpixel((0, 0), (255, 0, 0, 255))
    monkeypatch.setitem(
        sys.modules,
        "rembg",
        SimpleNamespace(remove=lambda image: foreground),
    )

    result_path = replace_background(
        str(source),
        str(output),
        background_color=(0, 255, 0),
    )

    with Image.open(result_path) as result:
        assert result.getpixel((0, 0)) == (255, 0, 0)
        assert result.getpixel((1, 1)) == (0, 255, 0)


def test_replace_background_with_image_fits_canvas(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    background = tmp_path / "background.png"
    output = tmp_path / "result.png"
    Image.new("RGB", (2, 2), (0, 0, 0)).save(source)
    Image.new("RGB", (4, 2), (20, 40, 60)).save(background)
    foreground = Image.new("RGBA", (2, 2), (255, 0, 0, 0))
    monkeypatch.setitem(
        sys.modules,
        "rembg",
        SimpleNamespace(remove=lambda image: foreground),
    )

    result_path = replace_background(
        str(source),
        str(output),
        background_image_path=str(background),
    )

    with Image.open(result_path) as result:
        assert result.size == (2, 2)
        assert result.getpixel((0, 0)) == (20, 40, 60)


def test_replace_background_requires_exactly_one_background(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (2, 2)).save(source)

    with pytest.raises(ValueError, match="tepat satu"):
        replace_background(str(source), str(tmp_path / "result.png"))
