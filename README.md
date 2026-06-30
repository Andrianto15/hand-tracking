# Hand Tracking Experiment

Aplikasi realtime untuk melacak (tracking) telapak tangan dan persendian jari menggunakan webcam secara langsung, divisualisasikan dengan titik landmark dan garis koneksi ("tulang") di atas video menggunakan **Google MediaPipe Tasks API** dan **OpenCV**.

## Fitur Utama

- 👐 **Multi-Hand Tracking**: Mendukung pelacakan hingga 2 tangan sekaligus secara realtime.
- 📐 **High-Tech Skeleton**: Visualisasi 21 titik landmark (persendian) dan tulang tangan dengan skema warna neon yang premium.
- ℹ️ **Interactive Info Overlay** (Dapat dinyalakan/dimatikan dengan menekan tombol `i`):
  - **FPS Counter**: Indikator performa frame per detik.
  - **Left/Right Hand Label**: Klasifikasi tangan kiri/kanan fisik yang presisi.
  - **Finger Labeling**: Menampilkan nama jari (`Thumb`, `Index`, `Middle`, `Ring`, `Pinky`) di ujung masing-masing jari.
  - **Joint Coordinates**: Menampilkan koordinat piksel `(x, y)` secara realtime langsung di setiap sendi.
- 🚀 **HD Resolution**: Default kamera diset ke 1280x720 (720p) untuk ketajaman optimal dan deteksi jarak jauh yang lebih akurat.

## Kontrol Keyboard

Fokus pada jendela video OpenCV untuk menggunakan kontrol berikut:

| Tombol | Fungsi |
|---|---|
| `i` | Toggle (on/off) seluruh overlay info tambahan |
| `q` | Keluar dari aplikasi dengan bersih |

---

## Persyaratan Sistem

- Python 3.8 ke atas
- Kamera/Webcam aktif
- macOS (direkomendasikan, mendukung macOS camera permissions)

## Cara Instalasi & Menjalankan

### 1. Setup Virtual Environment (Opsional namun disarankan)

```bash
# Membuat virtual environment
python3 -m venv venv

# Mengaktifkan venv (macOS/Linux)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Model File

MediaPipe Tasks API membutuhkan file model `.task` yang terpisah. Unduh menggunakan perintah berikut dan letakkan di root folder project:

```bash
curl -o hand_landmarker.task -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

### 4. Jalankan Aplikasi

```bash
python hand_tracker.py
```

---

## Struktur File Project

```text
hand-tracking/
├── venv/                 # Virtual environment
├── .gitignore            # File pengecualian Git
├── README.md             # Dokumentasi panduan
├── PRD.md                # Product Requirements Document
├── Implementation.md     # Rencana implementasi
├── requirements.txt      # Daftar dependency project
├── hand_landmarker.task  # File model MediaPipe (diabaikan oleh git)
└── hand_tracker.py       # Source code utama
```
