# AI Visual Inspector

A simple Python image processing project built with OpenCV for basic computer vision workflows. This project reads an image from `data/input`, applies a few transformations, and saves the results into `data/output`.

## Fitur

- Mengubah citra ke mode grayscale
- Mengubah ukuran gambar ke dimensi yang diinginkan
- Menerapkan Gaussian blur
- Mendeteksi tepi menggunakan Canny edge detection
- Menyimpan hasil ke folder output

## Struktur Proyek

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

## Persyaratan

Pastikan Python 3.10+ sudah terpasang, lalu install dependency:

```bash
pip install -r requirements.txt
```

## Cara Menjalankan

1. Letakkan file gambar yang ingin diproses di folder `data/input/`.
2. Jalankan program:

```bash
python app/main.py
```

atau

```bash
python -m app.main
```

3. Hasil pemrosesan akan otomatis disimpan ke folder `data/output/` dengan nama:
   - `original.jpg`
   - `grayscale.jpg`
   - `blurred.jpg`
   - `edges.jpg`

## Contoh Output

Program akan memproses gambar pertama yang ditemukan di `data/input/` dan menghasilkan versi yang sudah di-resize, diubah ke grayscale, diberi efek blur, dan dideteksi tepinya.

## Testing

Untuk menjalankan unit test:

```bash
pytest -q
```

## Dependensi

- `numpy`
- `opencv-python`
- `Pillow`
- `matplotlib`
- `pytest`

## Catatan

Project ini cocok untuk pembelajaran dasar pemrosesan citra dan deteksi tepi menggunakan OpenCV. Jika ingin dikembangkan lebih lanjut, fitur seperti thresholding, contour detection, atau klasifikasi objek bisa ditambahkan.
