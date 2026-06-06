import cv2
import os
import time
import numpy as np
import tkinter as tk
from tkinter import messagebox

from trainer import train_model   # called when user presses T

# ── emotion key map ───────────────────────────────
EMOTION_KEYS = {
    ord('h'): 'happy',
    ord('s'): 'sad',
    ord('a'): 'angry',
    ord('n'): 'neutral',
    ord('u'): 'surprise',
    ord('f'): 'fear',
}

DATASET_DIR = "dataset"
IMG_SIZE    = 64          # images saved as 64×64 px


# ──────────────────────────────────────────────────
def _ensure_dirs():
    """Create dataset sub-folders if they don't exist."""
    for name in EMOTION_KEYS.values():
        os.makedirs(os.path.join(DATASET_DIR, name), exist_ok=True)


def _count_images():
    """Return dict {emotion: count}."""
    counts = {}
    for name in EMOTION_KEYS.values():
        folder = os.path.join(DATASET_DIR, name)
        if os.path.isdir(folder):
            counts[name] = len([f for f in os.listdir(folder)
                                 if f.lower().endswith(('.png', '.jpg'))])
        else:
            counts[name] = 0
    return counts


def _draw_overlay(frame, counts, fps, last_label):
    """Draw instruction overlay and stats onto frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent dark panel on the left
    cv2.rectangle(overlay, (0, 0), (260, h), (15, 15, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Title
    cv2.putText(frame, "DATA COLLECTION", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 200, 255), 2)

    # Key bindings
    keys_info = [
        ("H", "Happy",    counts.get('happy',   0)),
        ("S", "Sad",      counts.get('sad',      0)),
        ("A", "Angry",    counts.get('angry',    0)),
        ("N", "Neutral",  counts.get('neutral',  0)),
        ("U", "Surprise", counts.get('surprise', 0)),
        ("F", "Fear",     counts.get('fear',     0)),
    ]
    y = 60
    for key, emo, cnt in keys_info:
        cv2.putText(frame, f"[{key}] {emo:<10} {cnt:>4}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (220, 220, 220), 1)
        y += 24

    cv2.putText(frame, "[T] Train Model", (10, y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 230, 80), 1)
    cv2.putText(frame, "[Q] Quit",        (10, y + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 160, 255), 1)

    # FPS (top-right)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 180), 2)

    # Last captured label feedback
    if last_label:
        cv2.putText(frame, f"Saved: {last_label.upper()}", (w // 2 - 70, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 100), 2)


# ──────────────────────────────────────────────────
def run_train_mode(status_var=None):
    """
    Open webcam, let user press emotion keys to collect face images.
    Press T to trigger model training.
    Press Q to quit.

    status_var: optional tk.StringVar for GUI status bar updates.
    """
    def _status(msg):
        if status_var is not None:
            try:
                status_var.set(msg)
            except Exception:
                pass
        print(msg)

    _ensure_dirs()

    # ── Haar Cascade face detector ─────────────────
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        _status("Error: Could not load Haar Cascade!")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        _status("Error: Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    counts     = _count_images()
    last_label = ""
    label_timer = 0
    prev_time  = time.time()
    fps        = 0.0

    _status("Train Mode: press emotion keys to capture. T=Train, Q=Quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            _status("Error: Failed to read from webcam.")
            break

        frame = cv2.flip(frame, 1)   # mirror for natural feel

        # ── FPS ──
        now  = time.time()
        fps  = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-9))
        prev_time = now

        # ── Face detection ──
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Histogram equalise for low-light robustness
        gray_eq = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(
            gray_eq, scaleFactor=1.1, minNeighbors=5,
            minSize=(60, 60))

        # Draw bounding boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 255), 2)

        # Clear label after ~1 s
        if label_timer and time.time() - label_timer > 1.0:
            last_label  = ""
            label_timer = 0

        _draw_overlay(frame, counts, fps, last_label)
        cv2.imshow("Train Mode — Facial Emotion Detection", frame)

        # ── Key handling ──
        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord('q'):
            _status("Train Mode closed by user.")
            break

        # Train
        if key == ord('t'):
            _status("Training model…  (this may take a few minutes)")
            cap.release()
            cv2.destroyAllWindows()
            train_model(status_callback=_status)
            _status("Training complete! Model saved as model.hdf5")
            return

        # Emotion capture
        if key in EMOTION_KEYS and len(faces) > 0:
            emotion = EMOTION_KEYS[key]
            # Use the largest detected face
            (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))

            # Save image
            folder   = os.path.join(DATASET_DIR, emotion)
            idx      = counts[emotion]
            filename = os.path.join(folder, f"{emotion}_{idx:05d}.png")
            cv2.imwrite(filename, face_roi)

            counts[emotion] += 1
            last_label       = emotion
            label_timer      = time.time()
            _status(f"Saved {emotion} #{counts[emotion]}")

        elif key in EMOTION_KEYS and len(faces) == 0:
            # Tell user no face found
            last_label  = "NO FACE DETECTED"
            label_timer = time.time()

    cap.release()
    cv2.destroyAllWindows()
