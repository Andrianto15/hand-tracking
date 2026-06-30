import cv2
import mediapipe as mp
import time
import os
import sys
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Bone Connection Hand Landmarks ---
# Define mapping connection hand landmarks (21 joints)
try:
    HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS
except Exception:
    # Fallback manual jika solutions tidak tersedia
    HAND_CONNECTIONS = {
        (0, 1), (1, 2), (2, 3), (3, 4),      # Jempol (Thumb)
        (0, 5), (5, 6), (6, 7), (7, 8),      # Telunjuk (Index)
        (9, 10), (10, 11), (11, 12),         # Tengah (Middle)
        (13, 14), (14, 15), (15, 16),         # Manis (Ring)
        (0, 17), (17, 18), (18, 19), (19, 20), # Kelingking (Pinky)
        (5, 9), (9, 13), (13, 17)            # Telapak (Palm)
    }

# Mapping ID landmark ke nama jari untuk label di ujung jari
FINGER_TIPS = {
    4: "Thumb",
    8: "Index",
    12: "Middle",
    16: "Ring",
    20: "Pinky"
}

def detect_finger_states(landmarks, handedness):
    """Mendeteksi status terangkat (True/False) untuk setiap jari."""
    # Helper untuk menghitung jarak Euclidean 2D (lebih stabil daripada 3D karena tidak ada noise depth)
    def get_dist_2d(p1, p2):
        return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5
        
    # Jari telunjuk (8 vs 6), tengah (12 vs 10), manis (16 vs 14), kelingking (20 vs 18)
    # y semakin kecil berarti posisi vertikal semakin tinggi (ke atas layar)
    index_raised = landmarks[8].y < landmarks[6].y
    middle_raised = landmarks[12].y < landmarks[10].y
    ring_raised = landmarks[16].y < landmarks[14].y
    pinky_raised = landmarks[20].y < landmarks[18].y
    
    # Ibu jari (Thumb): menggunakan rasio jarak 2D tip ke wrist vs mcp ke wrist
    tip = landmarks[4]
    mcp = landmarks[2]
    wrist = landmarks[0]
    dist_tip_wrist = get_dist_2d(tip, wrist)
    dist_mcp_wrist = get_dist_2d(mcp, wrist)
    
    if dist_mcp_wrist > 0:
        ratio = dist_tip_wrist / dist_mcp_wrist
    else:
        ratio = 0.0
        
    # Threshold default 1.3 (menggunakan 2D distance ini jauh lebih stabil)
    thumb_raised = ratio > 1.3
    
    return {
        "thumb": thumb_raised,
        "index": index_raised,
        "middle": middle_raised,
        "ring": ring_raised,
        "pinky": pinky_raised
    }, ratio

def classify_gesture(finger_states):
    """Mengklasifikasikan gesture berdasarkan status jari."""
    thumb = finger_states["thumb"]
    index = finger_states["index"]
    middle = finger_states["middle"]
    ring = finger_states["ring"]
    pinky = finger_states["pinky"]
    
    # 1. Semua 5 jari True -> "Open Palm"
    if thumb and index and middle and ring and pinky:
        return "Open Palm"
        
    # 2. Semua 5 jari False -> "Fist"
    if not (thumb or index or middle or ring or pinky):
        return "Fist"
        
    # 3. Hanya thumb=True, sisanya False -> "Thumbs Up"
    if thumb and not (index or middle or ring or pinky):
        return "Thumbs Up"
        
    # 4. index=True, middle=True, sisanya False -> "Peace"
    if index and middle and not (thumb or ring or pinky):
        return "Peace"
        
    # 5. Fallback: Hitung jumlah jari yang True -> "{n} Fingers"
    n = sum([thumb, index, middle, ring, pinky])
    return f"{n} Fingers"

def update_gesture_state(state, raw_gesture, debounce_frames=10):
    """Memperbarui state gesture dengan mekanisme debouncing."""
    if raw_gesture == state["candidate_gesture"]:
        state["candidate_count"] += 1
    else:
        state["candidate_gesture"] = raw_gesture
        state["candidate_count"] = 1
        
    if state["candidate_count"] >= debounce_frames and state["candidate_gesture"] != state["current_gesture"]:
        state["current_gesture"] = state["candidate_gesture"]
        return True
    return False

def draw_transparent_rect(img, pt1, pt2, color, alpha):
    """Menggambar rectangle transparan (glassmorphic) untuk UI premium."""
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_landmarks_and_bones(frame, hand_landmarks, show_coords):
    """Menggambar sendi (landmarks) dan tulang (connections)."""
    h, w, _ = frame.shape
    
    # 1. Gambar tulang (koneksi antar landmark)
    # BGR (255, 230, 0) = Electric Cyan/Aqua
    bone_color = (255, 230, 0)
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
            pt1 = (int(hand_landmarks[start_idx].x * w), int(hand_landmarks[start_idx].y * h))
            pt2 = (int(hand_landmarks[end_idx].x * w), int(hand_landmarks[end_idx].y * h))
            cv2.line(frame, pt1, pt2, bone_color, 2, cv2.LINE_AA)
            
    # 2. Gambar sendi (bulatan koordinat)
    # BGR (153, 51, 255) = Neon Purple/Rose
    joint_color_outer = (153, 51, 255)
    joint_color_inner = (255, 255, 255)
    
    for idx, lm in enumerate(hand_landmarks):
        cx, cy = int(lm.x * w), int(lm.y * h)
        
        # Outer circle (glow effect)
        cv2.circle(frame, (cx, cy), 6, joint_color_outer, -1, cv2.LINE_AA)
        # Inner dot (white center)
        cv2.circle(frame, (cx, cy), 2, joint_color_inner, -1, cv2.LINE_AA)
        
        # Gambar koordinat di dekat sendi jika opsi diaktifkan
        if show_coords:
            coord_text = f"({cx},{cy})"
            # Render drop shadow hitam agar kontras di background apapun
            cv2.putText(frame, coord_text, (cx + 8, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.3, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, coord_text, (cx + 8, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.3, (200, 200, 200), 1, cv2.LINE_AA)

def draw_info_overlay(frame, fps, show_info, handedness_list, hand_landmarks_list, gesture_states):
    """Menggambar UI Panel di pojok kiri atas dan label Left/Right/Fingers."""
    h, w, _ = frame.shape
    help_text = "Press 'i' to toggle info | 'q' to quit"
    
    if show_info:
        # Panel Detail
        panel_w = 260
        panel_h = 75 + max(1, len(hand_landmarks_list)) * 25
        
        # Background
        draw_transparent_rect(frame, (10, 10), (10 + panel_w, 10 + panel_h), (30, 30, 30), 0.7)
        # Border
        cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (80, 80, 80), 1, cv2.LINE_AA)
        
        # Teks Info
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Hands Detected: {len(hand_landmarks_list)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        y_offset = 85
        for i, hand_landmarks in enumerate(hand_landmarks_list):
            label = "Unknown"
            if handedness_list and i < len(handedness_list):
                # Koreksi label karena kamera dimirror (flip horizontal)
                raw_label = handedness_list[i][0].category_name
                label = "Left" if raw_label == "Right" else "Right"
            
            # Left/Right Label (Warna: Hijau untuk Right, Merah untuk Left)
            color = (100, 255, 100) if label == "Right" else (100, 100, 255)
            gesture = gesture_states[label]["current_gesture"] if label in gesture_states else None
            ratio = gesture_states[label]["thumb_ratio"] if (label in gesture_states and "thumb_ratio" in gesture_states[label]) else 0.0
            gesture_text = f" ({gesture}, T:{ratio:.2f})" if gesture else f" (None, T:{ratio:.2f})"
            
            cv2.putText(frame, f"Hand {i+1}: {label}{gesture_text}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            y_offset += 25
            
        cv2.putText(frame, help_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
        
        # 3. Gambar label Left/Right nempel di tangan dan nama jari di ujung jari
        for i, hand_landmarks in enumerate(hand_landmarks_list):
            if not hand_landmarks:
                continue
            
            # Ambil label Left/Right
            label = "Unknown"
            if handedness_list and i < len(handedness_list):
                # Koreksi label karena kamera dimirror (flip horizontal)
                raw_label = handedness_list[i][0].category_name
                label = "Left" if raw_label == "Right" else "Right"
            
            # Draw label di dekat wrist (landmark 0)
            wrist = hand_landmarks[0]
            cx, cy = int(wrist.x * w), int(wrist.y * h)
            color = (0, 255, 0) if label == "Right" else (0, 0, 255)
            
            cv2.putText(frame, label.upper(), (cx - 20, cy + 25), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, label.upper(), (cx - 20, cy + 25), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, color, 1, cv2.LINE_AA)
            
            # Draw gesture dan thumb ratio di bawah label Left/Right
            gesture = gesture_states[label]["current_gesture"] if label in gesture_states else None
            ratio = gesture_states[label]["thumb_ratio"] if (label in gesture_states and "thumb_ratio" in gesture_states[label]) else 0.0
            gesture_display = f"{gesture if gesture else 'None'} (T:{ratio:.2f})"
            cv2.putText(frame, gesture_display, (cx - 20, cy + 45), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, gesture_display, (cx - 20, cy + 45), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Draw nama jari di ujung jari
            for idx, name in FINGER_TIPS.items():
                if idx < len(hand_landmarks):
                    lm = hand_landmarks[idx]
                    fx, fy = int(lm.x * w), int(lm.y * h)
                    cv2.putText(frame, name, (fx - 15, fy - 15), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.4, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, name, (fx - 15, fy - 15), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.4, (0, 255, 150), 1, cv2.LINE_AA)
    else:
        # Panel Minimalis
        panel_w = 120
        panel_h = 45
        draw_transparent_rect(frame, (10, 10), (10 + panel_w, 10 + panel_h), (30, 30, 30), 0.7)
        cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (80, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

def main():
    model_path = 'hand_landmarker.task'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' tidak ditemukan.")
        print("Silakan jalankan:")
        print("curl -o hand_landmarker.task -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        sys.exit(1)
        
    print("Inisialisasi MediaPipe Hand Landmarker...")
    # Konfigurasi model Hand Landmarker
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Buka webcam
    print("Membuka webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Gagal membuka webcam. Silakan periksa camera permission di macOS.")
        sys.exit(1)
        
    # Set resolusi HD 720p (optimal untuk ketajaman, akurasi deteksi jarak jauh, & performa lancar di macOS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # State variables
    show_info = True
    prev_time = time.time()
    last_timestamp_ms = -1
    
    # Gesture state per hand
    gesture_states = {
        "Left": {
            "current_gesture": None,
            "candidate_gesture": None,
            "candidate_count": 0,
            "thumb_ratio": 0.0
        },
        "Right": {
            "current_gesture": None,
            "candidate_gesture": None,
            "candidate_count": 0,
            "thumb_ratio": 0.0
        }
    }
    
    print("\n--- Hand Tracker Berhasil Dijalankan ---")
    print("Tekan 'i' di window video untuk menyalakan/mematikan info overlay.")
    print("Tekan 'q' di window video untuk keluar.\n")
    
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Warning: Gagal membaca frame dari webcam. Skipping...")
                continue
                
            # Flip frame secara horizontal agar terasa natural seperti cermin
            frame = cv2.flip(frame, 1)
            
            # Hitung FPS
            current_time = time.time()
            time_diff = current_time - prev_time
            fps = 1.0 / time_diff if time_diff > 0 else 0.0
            prev_time = current_time
            
            # Konversi frame BGR ke RGB untuk MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Dapatkan timestamp monoton meningkat dalam milidetik
            timestamp_ms = int(current_time * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms
            
            # Deteksi
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # Ambil hand landmarks & handedness
            hand_landmarks_list = result.hand_landmarks
            handedness_list = result.handedness
            
            # Track detected hands in this frame to reset the other hand if it is not present
            detected_hands = set()
            
            for i, hand_landmarks in enumerate(hand_landmarks_list):
                if handedness_list and i < len(handedness_list):
                    # Koreksi label karena kamera dimirror (flip horizontal)
                    raw_label = handedness_list[i][0].category_name
                    hand_label = "Left" if raw_label == "Right" else "Right"
                    detected_hands.add(hand_label)
                    
                    # Step 1: Detect finger states and get ratio
                    finger_states, thumb_ratio = detect_finger_states(hand_landmarks, hand_label)
                    gesture_states[hand_label]["thumb_ratio"] = thumb_ratio
                    
                    # Step 2: Classify raw gesture
                    raw_gesture = classify_gesture(finger_states)
                    
                    # Step 4: Update gesture state (debounce)
                    state_changed = update_gesture_state(gesture_states[hand_label], raw_gesture)
                    
                    # Step 5: Logging
                    if state_changed:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"[{timestamp}] {hand_label} Hand: {gesture_states[hand_label]['current_gesture']}")
            
            # Reset state for undetected hands
            for hand_label in ["Left", "Right"]:
                if hand_label not in detected_hands:
                    gesture_states[hand_label] = {
                        "current_gesture": None,
                        "candidate_gesture": None,
                        "candidate_count": 0,
                        "thumb_ratio": 0.0
                    }
            
            # 1. Gambar skeleton & landmarks
            for hand_landmarks in hand_landmarks_list:
                # Koordinat detil digambar nempel di sendi hanya jika show_info = True
                draw_landmarks_and_bones(frame, hand_landmarks, show_coords=show_info)
                
            # 2. Gambar overlay info & status panel
            draw_info_overlay(frame, fps, show_info, handedness_list, hand_landmarks_list, gesture_states)
            
            # Tampilkan frame
            cv2.imshow("Hand Tracking Experiment", frame)
            
            # Keyboard listener
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Menutup aplikasi...")
                break
            elif key == ord('i'):
                show_info = not show_info
                print(f"Info overlay diubah menjadi: {show_info}")
                
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Cleanup selesai. Aplikasi keluar bersih.")

if __name__ == "__main__":
    main()
