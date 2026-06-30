# Implementation Plan: Gesture Recognition v2

Referensi: `PRD.md` section 7a (Gesture Recognition v2)

## Urutan Implementasi

### Step 1 — `detect_finger_states(landmarks, handedness)`

- Input: 21 landmark satu tangan + label Left/Right
- Untuk telunjuk, tengah, manis, kelingking:
  - Bandingkan `tip.y` vs `pip.y` → terangkat jika `tip.y < pip.y`
- Untuk ibu jari:
  - Hitung jarak euclidean `tip → wrist` dan `mcp → wrist`
  - Rasio = `dist(tip, wrist) / dist(mcp, wrist)`
  - Terangkat jika rasio > threshold (mulai dari `1.3`, perlu kalibrasi manual saat testing)
- Output: list/dict boolean, misal `{thumb: bool, index: bool, middle: bool, ring: bool, pinky: bool}`

**Test manual:** jalankan dengan buka/tutup tiap jari satu-satu, print hasil ke console, pastikan boolean berubah sesuai gerakan.

---

### Step 2 — `classify_gesture(finger_states)`

- Input: dict boolean dari Step 1
- Cek urutan prioritas:
  1. Semua 5 jari `True` → `"Open Palm"`
  2. Semua 5 jari `False` → `"Fist"`
  3. Hanya `thumb=True`, sisanya `False` → `"Thumbs Up"`
  4. `index=True, middle=True`, sisanya `False` → `"Peace"`
  5. Tidak match semua di atas → fallback hitung jumlah `True` → `"{n} Fingers"` (n = 0-5)

**Test manual:** coba tiap gesture target satu-satu, pastikan label yang muncul sesuai.

---

### Step 3 — `GestureState` (struktur data per tangan)

- Dict atau dataclass sederhana:

```python
{
  "current_gesture": None,
  "candidate_gesture": None,
  "candidate_count": 0
}
```

- Disimpan di `main()` sebagai dua instance: `gesture_state_left`, `gesture_state_right` (atau dict keyed by handedness)

---

### Step 4 — `update_gesture_state(state, raw_gesture, debounce_frames=10)`

- Logika:
  1. Jika `raw_gesture == state["candidate_gesture"]` → `candidate_count += 1`
  2. Jika beda → `candidate_gesture = raw_gesture`, `candidate_count = 1`
  3. Jika `candidate_count >= debounce_frames` dan `candidate_gesture != current_gesture`:
     - `current_gesture = candidate_gesture`
     - return `True` (artinya ada perubahan stabil → trigger log)
  4. Selain itu return `False`
- Jika tangan tidak terdeteksi di frame ini → reset state ke kondisi awal (`None`, `None`, `0`)

**Test manual:** goyangkan tangan cepat antar 2 gesture, pastikan label di layar tidak flicker (stabil ~0.3 detik baru ganti).

---

### Step 5 — Logging

- Saat `update_gesture_state()` return `True`:

```python
timestamp = datetime.now().strftime("%H:%M:%S")
print(f"[{timestamp}] {handedness} Hand: {state['current_gesture']}")
```

- Tambahkan `import datetime` di awal file

---

### Step 6 — Integrasi ke `main()` loop

- Setelah deteksi landmark per tangan tiap frame:
  1. Panggil `detect_finger_states()` → `classify_gesture()` → dapat `raw_gesture`
  2. Panggil `update_gesture_state()` dengan state tangan yang sesuai (Left/Right)
  3. Jika ada perubahan stabil → print log
  4. Render label gesture (`current_gesture`) ke frame, dekat posisi tangan (pakai `cv2.putText`, posisi bisa ambil dari wrist landmark)

---

### Step 7 — Testing menyeluruh

Checklist manual sebelum dianggap selesai:

- [ ] Semua 4 gesture spesifik terdeteksi benar (Open Palm, Fist, Thumbs Up, Peace)
- [ ] Fallback finger count (0-5) muncul saat gesture tidak match yang spesifik
- [ ] Label di layar tidak flicker, update halus setelah ~0.3 detik
- [ ] Log console muncul hanya saat gesture stabil berubah, tidak spam
- [ ] 2 tangan sekaligus punya state independen (Left beda gesture dari Right, tidak saling pengaruh)
- [ ] Tangan keluar dari frame → state reset, masuk lagi → deteksi ulang normal
- [ ] Toggle info (`i`) tetap berfungsi normal, tidak conflict dengan label gesture baru

---

## Catatan Implementasi

- Threshold ibu jari (`1.3`) kemungkinan perlu disesuaikan saat testing langsung — beda ukuran tangan/jarak ke kamera bisa mempengaruhi rasio jarak.
- `debounce_frames=10` asumsi capture berjalan ~30 FPS. Jika FPS jauh berbeda di environment kamu, mungkin perlu disesuaikan.
- Semua fungsi baru ditambahkan ke file yang sama (`hand_tracker.py`), sesuai arsitektur single-file di PRD.
