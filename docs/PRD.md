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

**In scope (v2 - Gesture Recognition):**

- Deteksi gesture statis rule-based geometris: Open Palm, Fist, Thumbs Up, Peace, fallback Finger Count (0-5)
- State machine debounce per tangan (independen Left/Right)
- Log gesture ke console saat berubah (stabil)
- Tampilan label gesture stabil di layar

**Out of scope:**

- Recording/save hasil video atau screenshot
- Support file video (hanya webcam realtime)
- UI selain OpenCV window (tidak ada GUI framework tambahan)
- Confidence indicator untuk gesture (di-drop dari scope eksperimen ini)
- Gesture dinamis (motion-based, misal swipe/wave)
- Machine learning classifier (masih rule-based geometris)

## 3. Tech Stack

- Python 3
- `opencv-python` — capture webcam, render overlay, window display
- `mediapipe` (Tasks API — Hand Landmarker) — deteksi tangan & 21 landmark per tangan

## 4. Arsitektur

Single file script: `hand_tracker.py`

| Fungsi                       | Tanggung Jawab                                                                                                                                      |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `main()`                     | Loop utama: capture frame → deteksi → gambar overlay → tampilkan window → handle keyboard                                                           |
| `draw_landmarks_and_bones()` | Gambar titik (circle) di tiap landmark + garis lurus antar sendi sesuai `HAND_CONNECTIONS`                                                          |
| `draw_info_overlay()`        | Gambar FPS, label jari, koordinat per titik, label Left/Right (semua toggle bareng)                                                                 |
| `calculate_fps()`            | Hitung FPS per frame                                                                                                                                |
| `detect_finger_states()`     | Cek per jari (4 jari pakai tip vs PIP, ibu jari pakai rasio jarak tip→wrist vs MCP→wrist) → return list boolean terangkat/tidak                     |
| `classify_gesture()`         | Dari finger states → tentukan gesture: cek urutan prioritas (Open Palm → Fist → Thumbs Up → Peace), fallback ke Finger Count (0-5) jika tidak match |
| `GestureState`               | Class/dict per tangan, simpan `current_gesture`, `candidate_gesture`, `candidate_count`                                                             |
| `update_gesture_state()`     | Update state machine debounce per tangan + trigger log saat `current_gesture` berubah stabil                                                        |

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

## 7a. Gesture Recognition (v2)

### Gesture yang Dideteksi (urutan prioritas cek)

1. **Open Palm** — 5 jari semua terangkat
2. **Fist** — 5 jari semua menggenggam (termasuk ibu jari dekat wrist)
3. **Thumbs Up** — cuma ibu jari terangkat, 4 jari lain menggenggam
4. **Peace (✌️)** — telunjuk + tengah terangkat, jari lain menggenggam
5. **Fallback: Finger Count (0-5)** — jumlah jari terangkat, dipakai jika tidak match gesture spesifik manapun

### Logika Deteksi per Jari

- **Telunjuk, tengah, manis, kelingking**: "terangkat" jika posisi y tip lebih kecil (lebih tinggi di frame) dari posisi y PIP joint-nya
- **Ibu jari**: hitung jarak euclidean tip→wrist dibanding jarak MCP joint ibu jari→wrist. Rasio di atas threshold tertentu → "terbuka/terangkat", di bawah threshold → "menggenggam"

### State Machine Debounce (per tangan, independen Left/Right)

State per tangan:

- `current_gesture` — gesture stabil yang sedang aktif (sudah di-log)
- `candidate_gesture` — gesture mentah yang baru terdeteksi di frame ini
- `candidate_count` — jumlah frame berturut-turut candidate ini muncul

Logika tiap frame:

1. Deteksi gesture mentah dari frame sekarang
2. Jika gesture mentah == `candidate_gesture` → `candidate_count` += 1
3. Jika beda → reset `candidate_gesture` = gesture baru, `candidate_count` = 1
4. Jika `candidate_count` >= 10 (≈0.3 detik @30fps) DAN `candidate_gesture` != `current_gesture` → update `current_gesture`, trigger log
5. Jika tangan hilang dari frame → reset state tangan tersebut

### Logging

Print ke console setiap kali `current_gesture` berubah (sudah lolos debounce):

```
[HH:MM:SS] Left Hand: Fist
[HH:MM:SS] Right Hand: 3 Fingers
```

Tidak disimpan ke file, console print saja.

### Tampilan di Layar

Label gesture ditampilkan dekat tangan (mengikuti style overlay info yang sudah ada), menggunakan `current_gesture` yang sudah stabil — bukan raw per-frame deteksi, sehingga label tidak flicker. Tidak ada confidence indicator yang ditampilkan.

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
- Gesture dinamis (motion-based, misal swipe/wave)
- Confidence indicator untuk gesture
- Machine learning classifier untuk gesture
- Log ke file (saat ini console print saja)
