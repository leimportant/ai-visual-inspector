try:
    import tkinter as tk
    from tkinter import filedialog, ttk
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Tkinter is required for the preview interface. Install python3-tk and try again."
    ) from exc

from pathlib import Path

import cv2
from PIL import Image, ImageTk

from app.core.image_processor import (
    apply_gaussian_blur,
    apply_threshold,
    detect_contours,
    detect_edges,
    draw_contours,
    resize_image,
    to_grayscale,
)
from app.utils.image_loader import load_image


def _to_pil_image(image):
    if len(image.shape) == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


class ImagePreviewApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Visual Inspector")
        self.root.geometry("1280x760")

        self.image_path = self._default_image_path()
        self.image = None

        self.width_var = tk.IntVar(value=640)
        self.kernel_var = tk.IntVar(value=5)
        self.threshold1_var = tk.IntVar(value=100)
        self.threshold2_var = tk.IntVar(value=200)
        self.threshold_value_var = tk.IntVar(value=127)
        self.max_value_var = tk.IntVar(value=255)
        self.contour_thickness_var = tk.IntVar(value=2)

        self._build_ui()
        self._load_image(self.image_path)

    def _default_image_path(self):
        input_dir = Path("data/input")
        candidates = sorted(path for path in input_dir.glob("*") if path.is_file())
        if not candidates:
            return None
        return candidates[0]

    def _build_ui(self):
        controls = ttk.Frame(self.root, padding=10)
        controls.pack(side="left", fill="y")

        ttk.Button(controls, text="Open Image", command=self.open_image).pack(fill="x", pady=(0, 8))
        ttk.Button(controls, text="Refresh Preview", command=self.refresh_preview).pack(fill="x", pady=(0, 12))

        for label, var, lower, upper in [
            ("Width", self.width_var, 50, 2000),
            ("Kernel Size", self.kernel_var, 1, 31),
            ("Threshold 1", self.threshold1_var, 0, 255),
            ("Threshold 2", self.threshold2_var, 0, 255),
            ("Threshold Value", self.threshold_value_var, 0, 255),
            ("Max Value", self.max_value_var, 0, 255),
            ("Contour Thickness", self.contour_thickness_var, 1, 10),
        ]:
            frame = ttk.Frame(controls)
            frame.pack(fill="x", pady=4)
            ttk.Label(frame, text=label, width=18).pack(side="left")
            ttk.Spinbox(
                frame,
                from_=lower,
                to=upper,
                textvariable=var,
                width=8,
            ).pack(side="right")

        preview_area = ttk.Frame(self.root)
        preview_area.pack(side="right", fill="both", expand=True)

        self.preview_frames = []
        for _ in range(6):
            panel = ttk.LabelFrame(preview_area, text="Preview", padding=5)
            panel.pack(side="left", fill="both", expand=True, padx=5, pady=5)
            label = ttk.Label(panel)
            label.pack(fill="both", expand=True)
            self.preview_frames.append((panel, label))

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff")])
        if not file_path:
            return
        self._load_image(Path(file_path))

    def _load_image(self, image_path):
        if image_path is None:
            return
        self.image_path = image_path
        self.image = load_image(str(image_path))
        self.refresh_preview()

    def refresh_preview(self):
        if self.image is None:
            return

        resized = resize_image(self.image, width=self.width_var.get())
        grayscale = to_grayscale(resized)
        blurred = apply_gaussian_blur(resized, kernel_size=self.kernel_var.get())
        edges = detect_edges(
            resized,
            threshold1=self.threshold1_var.get(),
            threshold2=self.threshold2_var.get(),
        )
        thresholded = apply_threshold(
            resized,
            threshold_value=self.threshold_value_var.get(),
            max_value=self.max_value_var.get(),
        )
        contours = detect_contours(
            resized,
            threshold_value=self.threshold_value_var.get(),
            max_value=self.max_value_var.get(),
        )
        contour_image = draw_contours(
            resized,
            contours,
            thickness=self.contour_thickness_var.get(),
        )

        processed = [
            ("Original", resized),
            ("Grayscale", grayscale),
            ("Blurred", blurred),
            ("Edges", edges),
            ("Thresholded", thresholded),
            ("Contours", contour_image),
        ]

        for (panel, label), (title, image_data) in zip(self.preview_frames, processed):
            panel.configure(text=title)
            photo = ImageTk.PhotoImage(_to_pil_image(image_data).resize((220, 180)))
            label.configure(image=photo)
            label.image = photo


def main():
    root = tk.Tk()
    ImagePreviewApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
