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
        self.root.geometry("1180x760")
        self.root.minsize(960, 640)

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
        self._configure_styles()
        shell = ttk.Frame(self.root, style="App.TFrame", padding=(18, 14))
        shell.pack(fill="both", expand=True)

        header = ttk.Frame(shell, style="App.TFrame")
        header.pack(fill="x", pady=(0, 12))
        ttk.Label(header, text="AI VISUAL INSPECTOR", style="Brand.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Ruang kerja gambar  /  Olah, buat, dan sempurnakan",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        notebook = ttk.Notebook(shell, style="Workspace.TNotebook")
        notebook.pack(fill="both", expand=True)

        preview_tab = ttk.Frame(notebook, style="Page.TFrame", padding=14)
        generation_tab = ttk.Frame(notebook, style="Page.TFrame", padding=14)
        background_tab = ttk.Frame(notebook, style="Page.TFrame", padding=14)
        notebook.add(preview_tab, text="  Pengolahan gambar  ")
        notebook.add(generation_tab, text="  Buat gambar  ")
        notebook.add(background_tab, text="  Ganti latar  ")

        controls = ttk.Frame(preview_tab, style="Card.TFrame", padding=14, width=238)
        controls.pack(side="left", fill="y")
        controls.pack_propagate(False)

        ttk.Label(controls, text="GAMBAR SUMBER", style="Section.TLabel").pack(
            anchor="w", pady=(0, 8)
        )
        ttk.Button(
            controls,
            text="Buka gambar",
            style="Primary.TButton",
            command=self.open_image,
        ).pack(fill="x", pady=(0, 6))
        ttk.Button(
            controls,
            text="Perbarui pratinjau",
            command=self.refresh_preview,
        ).pack(fill="x", pady=(0, 14))
        ttk.Separator(controls).pack(fill="x", pady=(0, 12))
        ttk.Label(controls, text="PENGATURAN PROSES", style="Section.TLabel").pack(
            anchor="w", pady=(0, 8)
        )

        for label, var, lower, upper in [
            ("Lebar", self.width_var, 50, 2000),
            ("Ukuran kernel", self.kernel_var, 1, 31),
            ("Ambang 1", self.threshold1_var, 0, 255),
            ("Ambang 2", self.threshold2_var, 0, 255),
            ("Nilai ambang", self.threshold_value_var, 0, 255),
            ("Nilai maksimum", self.max_value_var, 0, 255),
            ("Ketebalan kontur", self.contour_thickness_var, 1, 10),
        ]:
            frame = ttk.Frame(controls, style="Card.TFrame")
            frame.pack(fill="x", pady=3)
            ttk.Label(frame, text=label, style="Control.TLabel").pack(side="left")
            ttk.Spinbox(
                frame,
                from_=lower,
                to=upper,
                textvariable=var,
                width=7,
                style="Compact.TSpinbox",
            ).pack(side="right")

        preview_area = ttk.Frame(preview_tab, style="Page.TFrame", padding=(14, 0, 0, 0))
        preview_area.pack(side="right", fill="both", expand=True)
        ttk.Label(preview_area, text="Hasil pengolahan", style="PageTitle.TLabel").pack(
            anchor="w", pady=(0, 10)
        )

        self.preview_frames = []
        preview_grid = ttk.Frame(preview_area, style="Page.TFrame")
        preview_grid.pack(fill="both", expand=True)
        for index in range(6):
            panel = ttk.LabelFrame(
                preview_grid,
                text="Preview",
                style="Card.TLabelframe",
                padding=6,
            )
            panel.grid(
                row=index // 3,
                column=index % 3,
                sticky="nsew",
                padx=(0 if index % 3 == 0 else 6, 0),
                pady=(0 if index < 3 else 8, 0),
            )
            label = ttk.Label(panel, style="Preview.TLabel", anchor="center")
            label.pack(fill="both", expand=True)
            self.preview_frames.append((panel, label))
        for column in range(3):
            preview_grid.columnconfigure(column, weight=1, uniform="preview")
        for row in range(2):
            preview_grid.rowconfigure(row, weight=1, uniform="preview")

        self._build_generation_ui(generation_tab)
        self._build_background_ui(background_tab)

    def _configure_styles(self):
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 9), foreground="#1D2939")
        style.configure("App.TFrame", background="#F3F6FA")
        style.configure("Page.TFrame", background="#F3F6FA")
        style.configure("Card.TFrame", background="#FFFFFF")
        style.configure("TLabel", background="#F3F6FA", foreground="#1D2939")
        style.configure("Brand.TLabel", font=("Segoe UI", 15, "bold"), foreground="#17345C")
        style.configure("Muted.TLabel", foreground="#667085")
        style.configure("PageTitle.TLabel", font=("Segoe UI", 13, "bold"), foreground="#17345C")
        style.configure(
            "Section.TLabel",
            background="#FFFFFF",
            foreground="#667085",
            font=("Segoe UI", 8, "bold"),
        )
        style.configure("Control.TLabel", background="#FFFFFF", foreground="#344054")
        style.configure("Preview.TLabel", background="#E9EEF5", foreground="#667085")
        style.configure("TButton", padding=(9, 6), background="#FFFFFF")
        style.map("TButton", background=[("active", "#EAF0F8")])
        style.configure(
            "Primary.TButton",
            padding=(9, 7),
            background="#245A9B",
            foreground="#FFFFFF",
            font=("Segoe UI", 9, "bold"),
        )
        style.map("Primary.TButton", background=[("active", "#1D477B")])
        style.configure(
            "Workspace.TNotebook",
            background="#F3F6FA",
            borderwidth=0,
            tabmargins=(0, 0, 0, 0),
        )
        style.configure(
            "Workspace.TNotebook.Tab",
            padding=(12, 8),
            background="#E8EDF4",
            foreground="#475467",
            font=("Segoe UI", 9, "bold"),
        )
        style.map(
            "Workspace.TNotebook.Tab",
            background=[("selected", "#FFFFFF")],
            foreground=[("selected", "#245A9B")],
        )
        style.configure("Card.TLabelframe", background="#FFFFFF", bordercolor="#D9E1EA")
        style.configure(
            "Card.TLabelframe.Label",
            background="#FFFFFF",
            foreground="#344054",
            font=("Segoe UI", 9, "bold"),
        )
        style.configure("Compact.TSpinbox", padding=(4, 3))

    def _build_generation_ui(self, parent):
        ttk.Label(parent, text="Buat gambar", style="PageTitle.TLabel").pack(
            anchor="w", pady=(0, 3)
        )
        ttk.Label(
            parent,
            text="Jelaskan objek dan gaya gambar yang ingin dibuat.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(0, 12))

        body = ttk.Frame(parent, style="Page.TFrame")
        body.pack(fill="both", expand=True)
        prompt_panel = ttk.Frame(body, style="Card.TFrame", padding=14, width=320)
        prompt_panel.pack(side="left", fill="y")
        prompt_panel.pack_propagate(False)
        ttk.Label(prompt_panel, text="DESKRIPSI GAMBAR", style="Section.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            prompt_panel,
            text="Contoh: kelinci di taman atau motor matic berwarna merah.",
            style="Muted.TLabel",
            wraplength=285,
            justify="left",
        ).pack(anchor="w", pady=(5, 10))
        self.prompt_input = tk.Text(
            prompt_panel,
            height=5,
            wrap="word",
            relief="solid",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground="#D0D5DD",
            highlightcolor="#3974B8",
            padx=8,
            pady=7,
            font=("Segoe UI", 10),
        )
        self.prompt_input.pack(fill="x", pady=(0, 10))

        actions = ttk.Frame(prompt_panel, style="Card.TFrame")
        actions.pack(fill="x")
        self.generate_button = ttk.Button(
            actions,
            text="Buat gambar",
            style="Primary.TButton",
            command=self._generate_from_prompt,
        )
        self.generate_button.pack(fill="x")
        self.generation_status = ttk.Label(
            prompt_panel,
            text="Siap",
            style="Muted.TLabel",
            wraplength=285,
        )
        self.generation_status.pack(anchor="w", pady=(9, 0))

        result_panel = ttk.LabelFrame(
            body, text="  GENERATED IMAGE  ", style="Card.TLabelframe", padding=8
        )
        result_panel.pack(side="right", fill="both", expand=True, padx=(12, 0))
        self.generated_image_label = ttk.Label(
            result_panel,
            text="Your generated image will appear here.",
            style="Preview.TLabel",
            anchor="center",
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
        ttk.Label(parent, text="Ganti background", style="PageTitle.TLabel").pack(
            anchor="w", pady=(0, 12)
        )
        body = ttk.Frame(parent, style="Page.TFrame")
        body.pack(fill="both", expand=True)
        controls = ttk.Frame(body, style="Card.TFrame", padding=14, width=350)
        controls.pack(side="left", fill="y")
        controls.pack_propagate(False)
        ttk.Label(controls, text="GAMBAR SUMBER", style="Section.TLabel").pack(anchor="w")
        ttk.Entry(
            controls,
            textvariable=self.background_source_var,
            state="readonly",
        ).pack(fill="x", pady=(6, 5))
        ttk.Button(
            controls,
            text="Pilih gambar sumber",
            command=self._choose_background_source,
        ).pack(fill="x", pady=(0, 12))
        ttk.Separator(controls).pack(fill="x", pady=(0, 12))
        ttk.Label(controls, text="JENIS BACKGROUND", style="Section.TLabel").pack(anchor="w")

        ttk.Radiobutton(
            controls,
            text="Warna solid",
            variable=self.background_mode_var,
            value="color",
        ).pack(anchor="w", pady=(8, 4))

        color_row = ttk.Frame(controls, style="Card.TFrame")
        color_row.pack(fill="x", padx=(20, 0), pady=(0, 8))
        ttk.Button(
            color_row,
            text="Pilih warna",
            command=self._choose_background_color,
        ).pack(side="left")
        ttk.Label(
            color_row,
            textvariable=self.background_color_name_var,
        ).pack(side="left", padx=8)

        ttk.Radiobutton(
            controls,
            text="Gunakan gambar lain",
            variable=self.background_mode_var,
            value="image",
        ).pack(anchor="w", pady=(4, 4))

        background_row = ttk.Frame(controls, style="Card.TFrame")
        background_row.pack(fill="x", padx=(20, 0))
        ttk.Entry(
            background_row,
            textvariable=self.background_image_var,
            state="readonly",
        ).pack(fill="x", pady=(0, 5))
        ttk.Button(
            controls,
            text="Pilih gambar background",
            command=self._choose_background_image,
        ).pack(fill="x")

        actions = ttk.Frame(controls, style="Card.TFrame")
        actions.pack(fill="x", pady=(16, 0))
        self.background_button = ttk.Button(
            actions,
            text="Ganti background",
            style="Primary.TButton",
            command=self._replace_selected_background,
        )
        self.background_button.pack(fill="x")
        self.background_status = ttk.Label(
            controls, text="Siap", style="Muted.TLabel", wraplength=315
        )
        self.background_status.pack(anchor="w", pady=(9, 0))

        result_panel = ttk.LabelFrame(
            body, text="  PRATINJAU HASIL  ", style="Card.TLabelframe", padding=8
        )
        result_panel.pack(side="right", fill="both", expand=True, padx=(12, 0))
        self.background_result_label = ttk.Label(
            result_panel,
            text="Hasil penggantian background akan tampil di sini.",
            style="Preview.TLabel",
            anchor="center",
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
            ("Asli", resized),
            ("Grayscale", grayscale),
            ("Blur", blurred),
            ("Tepi", edges),
            ("Threshold", thresholded),
            ("Kontur", contour_image),
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
