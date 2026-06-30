# PRD: Hand Tracking Eksperimen (Python)

## 1. Latar Belakang & Tujuan

Eksperimen sederhana untuk melacak (tracking) telapak tangan secara realtime menggunakan webcam, lalu memvisualisasikan titik-titik persendian (landmark) dan garis lurus penghubung antar sendi (representasi "tulang") di atas video.

## 2. Scope

**In scope:**

- Tracking 2 tangan sekaligus secara realtime via webcam
- Visualisasi 21 titik landmark per tangan + garis koneksi antar titik
- Info tambahan on-screen: FPS, label nama jari, koordinat (x,y) tiap titik, label Left/Right
- Toggle satu tombol untuk on/off semua info tambahan
- Target platform: macOS

**Out of scope:**

- Recording/save hasil video atau screenshot
- Support file video (hanya webcam realtime)
- Gesture recognition / interpretasi gerakan tangan
- UI selain OpenCV window (tidak ada GUI framework tambahan)

## 3. Tech Stack

- Python 3
- `opencv-python` — capture webcam, render overlay, window display
- `mediapipe` (Tasks API — Hand Landmarker) — deteksi tangan & 21 landmark per tangan

## 4. Arsitektur

Single file script: `hand_tracker.py`

| Fungsi                       | Tanggung Jawab                                                                             |
| ---------------------------- | ------------------------------------------------------------------------------------------ |
| `main()`                     | Loop utama: capture frame → deteksi → gambar overlay → tampilkan window → handle keyboard  |
| `draw_landmarks_and_bones()` | Gambar titik (circle) di tiap landmark + garis lurus antar sendi sesuai `HAND_CONNECTIONS` |
| `draw_info_overlay()`        | Gambar FPS, label jari, koordinat per titik, label Left/Right (semua toggle bareng)        |
| `calculate_fps()`            | Hitung FPS per frame                                                                       |

## 5. Data Flow

1. OpenCV ambil frame dari webcam (`cv2.VideoCapture(0)`)
2. Frame dikirim ke MediaPipe Hand Landmarker → output: list landmark (x, y, z normalized) per tangan terdeteksi (maks 2), plus label Left/Right
3. Koordinat normalized dikonversi ke pixel (`x * width`, `y * height`)
4. Overlay digambar di atas frame asli:
   - Titik sendi → `cv2.circle`
   - Garis tulang → `cv2.line`
   - Teks info → `cv2.putText`
5. Frame ditampilkan via `cv2.imshow()`

## 6. Kontrol Keyboard

| Tombol | Aksi                                                                              |
| ------ | --------------------------------------------------------------------------------- |
| `q`    | Keluar program                                                                    |
| `i`    | Toggle semua info tambahan (FPS, label jari, koordinat, Left/Right) on/off bareng |

## 7. Tampilan Info Tambahan

- FPS counter
- Label nama jari (thumb, index, middle, ring, pinky)
- Koordinat (x, y) tiap titik sendi — ditampilkan kecil, nempel langsung di dekat tiap titik di video
- Jumlah tangan terdeteksi & label Left/Right per tangan

## 8. Error Handling

- Webcam gagal dibuka → print pesan error jelas, exit program
- Frame gagal dibaca di tengah loop → skip frame, lanjut loop (tidak crash)
- Library belum terinstall → pesan instruksi `pip install opencv-python mediapipe`

## 9. Dependencies

```
opencv-python
mediapipe
```

## 10. Out of Scope / Future Ideas (tidak dikerjakan sekarang)

- Save/record hasil ke file
- Support input dari file video
- Gesture recognition
