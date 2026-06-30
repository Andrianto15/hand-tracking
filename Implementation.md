# Implementation Plan: Hand Tracking Eksperimen

Referensi: `PRD.md`

## Step 1 — Setup Environment

Disarankan pakai virtual environment biar dependency gak nyampur sama project lain.

```bash
# buat folder project (kalau belum ada)
mkdir hand-tracking-experiment
cd hand-tracking-experiment

# buat virtual environment
python3 -m venv venv

# aktifkan venv (macOS)
source venv/bin/activate
```

## Step 2 — Install Dependencies

```bash
pip install opencv-python mediapipe
```

Cek instalasi sukses:

```bash
python3 -c "import cv2; import mediapipe; print('OK', cv2.__version__, mediapipe.__version__)"
```

## Step 3 — Download Model Hand Landmarker

MediaPipe Tasks API butuh file model `.task` yang didownload terpisah (bukan bagian dari pip package).

```bash
curl -o hand_landmarker.task -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

File ini diletakkan sejajar dengan `hand_tracker.py` di root folder project.

## Step 4 — Struktur File

```
hand-tracking-experiment/
├── venv/
├── hand_landmarker.task
└── hand_tracker.py
```

## Step 5 — Tulis Kode `hand_tracker.py`

Implementasi dipecah sesuai desain PRD: `main()`, `draw_landmarks_and_bones()`, `draw_info_overlay()`, `calculate_fps()`.

Urutan kerja saat coding:

1. Setup MediaPipe Hand Landmarker (load model, konfigurasi `num_hands=2`, mode `VIDEO` atau `LIVE_STREAM`)
2. Buka webcam dengan `cv2.VideoCapture(0)`
3. Loop: baca frame → convert BGR ke RGB → jalankan deteksi → ambil hasil landmark
4. Gambar titik & garis (`draw_landmarks_and_bones`)
5. Gambar info tambahan kalau toggle aktif (`draw_info_overlay`)
6. Tampilkan frame, baca input keyboard (`q` keluar, `i` toggle info)
7. Cleanup: release webcam, destroy window

## Step 6 — Cara Menjalankan

```bash
python3 hand_tracker.py
```

- Tekan `i` untuk toggle info tambahan (FPS, label jari, koordinat, Left/Right) on/off
- Tekan `q` untuk keluar

## Step 7 — Verifikasi / Testing Manual

Checklist yang dicek setelah jalan:

- [ ] Webcam terbuka, video muncul tanpa lag berlebihan
- [ ] Titik landmark muncul di tiap sendi saat tangan terdeteksi
- [ ] Garis lurus antar sendi tergambar sesuai struktur tulang tangan
- [ ] Tracking jalan untuk 1 tangan maupun 2 tangan sekaligus
- [ ] FPS counter muncul dan update tiap frame
- [ ] Label jari (thumb, index, dst) muncul dengan benar
- [ ] Koordinat (x,y) muncul nempel di tiap titik
- [ ] Label Left/Right sesuai tangan yang terdeteksi
- [ ] Tombol `i` berhasil toggle semua info bareng
- [ ] Tombol `q` keluar program dengan bersih (tidak hang/crash)
- [ ] Kalau webcam gak ada/gagal dibuka → muncul pesan error jelas, bukan crash traceback

## Step 8 — Known Issues / Troubleshooting

| Masalah                          | Kemungkinan Penyebab                       | Solusi                                                                                   |
| -------------------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: mediapipe` | venv belum aktif / belum install           | jalankan ulang Step 2, pastikan venv aktif                                               |
| Webcam tidak terbuka             | index kamera salah, atau permission macOS  | coba `cv2.VideoCapture(1)`, cek System Settings → Privacy → Camera, izinkan Terminal/IDE |
| FPS rendah / lag                 | resolusi webcam terlalu besar              | set resolusi lebih kecil via `cap.set(cv2.CAP_PROP_FRAME_WIDTH/HEIGHT, ...)`             |
| Model file not found             | `hand_landmarker.task` belum ada di folder | ulangi Step 3                                                                            |
