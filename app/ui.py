import queue
import threading
from datetime import datetime
try:
    import tkinter as tk
    from tkinter import colorchooser, filedialog, messagebox, ttk
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
from app.core.image_generator import ImageGenerator
from app.core.background_editor import replace_background
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
        self.image_generator = ImageGenerator()
        self.generation_results = queue.Queue()
        self.background_results = queue.Queue()
        self.background_source_var = tk.StringVar()
        self.background_image_var = tk.StringVar()
        self.background_mode_var = tk.StringVar(value="color")
        self.background_color = (255, 255, 255)
        self.background_color_name_var = tk.StringVar(value="Putih")

        self.width_var = tk.IntVar(value=640)
        self.kernel_var = tk.IntVar(value=5)
        self.threshold1_var = tk.IntVar(value=100)
        self.threshold2_var = tk.IntVar(value=200)
        self.threshold_value_var = tk.IntVar(value=127)
        self.max_value_var = tk.IntVar(value=255)
        self.contour_thickness_var = tk.IntVar(value=2)

        self._build_ui()
        self._load_image(self.image_path)
        self.root.after(100, self._check_generation_result)

    def _default_image_path(self):
        input_dir = Path("data/input")
        candidates = sorted(path for path in input_dir.glob("*") if path.is_file())
        if not candidates:
            return None
        return candidates[0]

    def _build_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)

        preview_tab = ttk.Frame(notebook)
        generation_tab = ttk.Frame(notebook)
        background_tab = ttk.Frame(notebook)
        notebook.add(preview_tab, text="Preview Pengolahan")
        notebook.add(generation_tab, text="Buat Gambar")
        notebook.add(background_tab, text="Ganti Background")

        controls = ttk.Frame(preview_tab, padding=10)
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

        preview_area = ttk.Frame(preview_tab)
        preview_area.pack(side="right", fill="both", expand=True)

        self.preview_frames = []
        for _ in range(6):
            panel = ttk.LabelFrame(preview_area, text="Preview", padding=5)
            panel.pack(side="left", fill="both", expand=True, padx=5, pady=5)
            label = ttk.Label(panel)
            label.pack(fill="both", expand=True)
            self.preview_frames.append((panel, label))

        self._build_generation_ui(generation_tab)
        self._build_background_ui(background_tab)

    def _build_generation_ui(self, parent):
        content = ttk.Frame(parent, padding=24)
        content.pack(fill="both", expand=True)

        ttk.Label(
            content,
            text="Deskripsikan gambar yang ingin dibuat",
            font=("", 16, "bold"),
        ).pack(anchor="w", pady=(0, 8))
        ttk.Label(
            content,
            text="Contoh: buatkan gambar sepeda motor atau buatkan gambar bunga.",
        ).pack(anchor="w", pady=(0, 8))

        self.prompt_input = tk.Text(content, height=4, wrap="word")
        self.prompt_input.pack(fill="x", pady=(0, 12))

        actions = ttk.Frame(content)
        actions.pack(fill="x")
        self.generate_button = ttk.Button(
            actions,
            text="Buat Gambar",
            command=self._generate_from_prompt,
        )
        self.generate_button.pack(side="left")
        self.generation_status = ttk.Label(actions, text="")
        self.generation_status.pack(side="left", padx=12)

        result_panel = ttk.LabelFrame(content, text="Hasil", padding=8)
        result_panel.pack(fill="both", expand=True, pady=(16, 0))
        self.generated_image_label = ttk.Label(
            result_panel,
            text="Gambar yang dibuat akan tampil di sini.",
        )
        self.generated_image_label.pack(fill="both", expand=True)

    def _generate_from_prompt(self):
        prompt = self.prompt_input.get("1.0", "end").strip()
        if not prompt:
            self.generation_status.configure(text="Masukkan deskripsi gambar.")
            return

        self.generate_button.state(["disabled"])
        self.generation_status.configure(text="Sedang membuat gambar...")
        threading.Thread(
            target=self._generate_image,
            args=(prompt,),
            daemon=True,
        ).start()

    def _generate_image(self, prompt):
        try:
            image = self.image_generator.generate(prompt)
            output_dir = Path("data/generated")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"generated_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
            image.save(output_path)
            self.generation_results.put((output_path, None))
        except Exception as exc:
            self.generation_results.put((None, str(exc)))

    def _check_generation_result(self):
        try:
            output_path, error = self.generation_results.get_nowait()
        except queue.Empty:
            pass
        else:
            self.generate_button.state(["!disabled"])
            if error:
                self.generation_status.configure(text="Gambar gagal dibuat.")
                messagebox.showerror("Gagal membuat gambar", error, parent=self.root)
            else:
                self.background_source_var.set(str(output_path))
                with Image.open(output_path) as saved_image:
                    image = saved_image.copy()
                image.thumbnail((900, 520))
                photo = ImageTk.PhotoImage(image)
                self.generated_image_label.configure(image=photo, text="")
                self.generated_image_label.image = photo
                self.generation_status.configure(text=f"Disimpan di {output_path}")

        try:
            output_path, error = self.background_results.get_nowait()
        except queue.Empty:
            pass
        else:
            self.background_button.state(["!disabled"])
            if error:
                self.background_status.configure(text="Background gagal diganti.")
                messagebox.showerror("Gagal mengganti background", error, parent=self.root)
            else:
                with Image.open(output_path) as saved_image:
                    image = saved_image.copy()
                image.thumbnail((900, 520))
                photo = ImageTk.PhotoImage(image)
                self.background_result_label.configure(image=photo, text="")
                self.background_result_label.image = photo
                self.background_status.configure(text=f"Disimpan di {output_path}")

        self.root.after(100, self._check_generation_result)

    def _build_background_ui(self, parent):
        content = ttk.Frame(parent, padding=24)
        content.pack(fill="both", expand=True)

        ttk.Label(
            content,
            text="Ganti background gambar",
            font=("", 16, "bold"),
        ).pack(anchor="w", pady=(0, 12))

        source_row = ttk.Frame(content)
        source_row.pack(fill="x", pady=4)
        ttk.Label(source_row, text="Gambar sumber", width=18).pack(side="left")
        ttk.Entry(
            source_row,
            textvariable=self.background_source_var,
            state="readonly",
        ).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(
            source_row,
            text="Pilih Gambar",
            command=self._choose_background_source,
        ).pack(side="left")

        ttk.Radiobutton(
            content,
            text="Warna solid",
            variable=self.background_mode_var,
            value="color",
        ).pack(anchor="w", pady=(12, 4))

        color_row = ttk.Frame(content)
        color_row.pack(fill="x", padx=(24, 0))
        ttk.Button(
            color_row,
            text="Pilih Warna",
            command=self._choose_background_color,
        ).pack(side="left")
        ttk.Label(
            color_row,
            textvariable=self.background_color_name_var,
        ).pack(side="left", padx=8)

        ttk.Radiobutton(
            content,
            text="Gambar lain",
            variable=self.background_mode_var,
            value="image",
        ).pack(anchor="w", pady=(12, 4))

        background_row = ttk.Frame(content)
        background_row.pack(fill="x", padx=(24, 0))
        ttk.Entry(
            background_row,
            textvariable=self.background_image_var,
            state="readonly",
        ).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(
            background_row,
            text="Pilih Background",
            command=self._choose_background_image,
        ).pack(side="left")

        actions = ttk.Frame(content)
        actions.pack(fill="x", pady=16)
        self.background_button = ttk.Button(
            actions,
            text="Ganti Background",
            command=self._replace_selected_background,
        )
        self.background_button.pack(side="left")
        self.background_status = ttk.Label(actions, text="")
        self.background_status.pack(side="left", padx=12)

        result_panel = ttk.LabelFrame(content, text="Hasil", padding=8)
        result_panel.pack(fill="both", expand=True)
        self.background_result_label = ttk.Label(
            result_panel,
            text="Hasil penggantian background akan tampil di sini.",
        )
        self.background_result_label.pack(fill="both", expand=True)

    def _choose_background_source(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff")]
        )
        if file_path:
            self.background_source_var.set(file_path)

    def _choose_background_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff")]
        )
        if file_path:
            self.background_image_var.set(file_path)

    def _choose_background_color(self):
        selected = colorchooser.askcolor(
            color="#%02x%02x%02x" % self.background_color,
            parent=self.root,
        )
        if selected[0] is not None:
            self.background_color = tuple(int(channel) for channel in selected[0])
            self.background_color_name_var.set(selected[1])

    def _replace_selected_background(self):
        image_path = self.background_source_var.get()
        if not image_path:
            self.background_status.configure(text="Pilih gambar sumber terlebih dahulu.")
            return

        mode = self.background_mode_var.get()
        background_image_path = self.background_image_var.get()
        if mode == "image" and not background_image_path:
            self.background_status.configure(text="Pilih gambar background terlebih dahulu.")
            return

        color = self.background_color if mode == "color" else None
        image = background_image_path if mode == "image" else None
        self.background_button.state(["disabled"])
        self.background_status.configure(text="Sedang mengganti background...")
        threading.Thread(
            target=self._replace_background,
            args=(image_path, color, image),
            daemon=True,
        ).start()

    def _replace_background(self, image_path, color, background_image_path):
        try:
            output_dir = Path("data/backgrounds")
            output_path = output_dir / (
                f"background_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
            )
            result_path = replace_background(
                image_path=image_path,
                output_path=str(output_path),
                background_color=color,
                background_image_path=background_image_path,
            )
            self.background_results.put((result_path, None))
        except Exception as exc:
            self.background_results.put((None, str(exc)))

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff")])
        if not file_path:
            return
        self._load_image(Path(file_path))

    def _load_image(self, image_path):
        if image_path is None:
            return
        self.image_path = image_path
        self.background_source_var.set(str(image_path))
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
