import cv2
import os
import csv
import json
import time
import numpy as np
from datetime import datetime

MODEL_PATH  = "model.keras"
MODEL_PATH2 = "model.keras"   # fallback newer format
LABEL_MAP   = "label_map.json"
LOG_FILE    = "emotion_log.csv"
IMG_SIZE    = 48

# Emotion → (BGR colour for bounding box)
EMOTION_COLORS = {
    'happy':    (0,   230, 118),
    'sad':      (255, 160,  60),
    'angry':    (60,  60,  255),
    'neutral':  (200, 200, 200),
    'surprise': (0,   220, 255),
    'fear':     (200,  60, 200),
}
DEFAULT_COLOR = (100, 200, 255)

# Emoji overlay (drawn as text)
EMOTION_EMOJI = {
    'happy':    ":-)",
    'sad':      ":-(",
    'angry':    ">:-(",
    'neutral':  ":-|",
    'surprise': ":-O",
    'fear':     "D-:",
}


# ──────────────────────────────────────────────────
def _load_model():
    """Load Keras model; tries .hdf5 then .keras."""
    import tensorflow as tf
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    if os.path.exists(MODEL_PATH2):
        return tf.keras.models.load_model(MODEL_PATH2)
    raise FileNotFoundError(
        "No trained model found (model.keras / model.keras).\n"
        "Please run Train Mode first and press T to train.")


def _load_labels():
    """Load label index→name mapping from label_map.json."""
    if os.path.exists(LABEL_MAP):
        with open(LABEL_MAP) as f:
            raw = json.load(f)
        # keys are strings in JSON
        return {int(k): v for k, v in raw.items()}
    # Fallback: alphabetical default
    default = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    return {i: e for i, e in enumerate(default)}


def _init_log():
    """Initialise CSV log file with headers if it doesn't exist."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "emotion", "confidence_%"])


def _log_prediction(emotion: str, confidence: float):
    """Append one prediction row to the CSV log."""
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            emotion,
            f"{confidence*100:.1f}"])


def _draw_bar(frame, x, y, w, emotion, confidence, color):
    """
    Draw a confidence bar below the bounding box.
    """
    bar_w = int(w * confidence)
    bar_h = 6
    cv2.rectangle(frame, (x, y), (x + w, y + bar_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), color, -1)


def _draw_hud(frame, fps: float, face_count: int):
    """Draw top-right HUD: FPS + face count."""
    h, w = frame.shape[:2]
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (w - 130, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 180), 2)
    cv2.putText(frame, f"Faces: {face_count}",
                (w - 130, 58), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1)
    cv2.putText(frame, "Q = Quit",
                (w - 130, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (130, 130, 130), 1)


# ──────────────────────────────────────────────────
def run_use_mode(status_var=None):
    """
    Open webcam, detect faces, predict emotion in real-time.
    Results are drawn on frame and logged to emotion_log.csv.
    Press Q to quit.

    status_var: optional tk.StringVar for GUI status bar.
    """
    def _status(msg):
        if status_var is not None:
            try:
                status_var.set(msg)
            except Exception:
                pass
        print(msg)

    # ── Load assets ──
    _status("Loading model…")
    try:
        model = _load_model()
    except FileNotFoundError as e:
        _status(str(e))
        return

    labels = _load_labels()
    _status(f"Model loaded. Labels: {labels}")

    # Haar cascade
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade  = cv2.CascadeClassifier(cascade_path)

    _init_log()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        _status("Error: Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time  = time.time()
    fps        = 0.0
    log_cooldown = {}   # emotion → last log time, to avoid flood
    LOG_INTERVAL = 2.0  # seconds between logging same emotion

    _status("Use Mode active. Press Q in the webcam window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            _status("Error: webcam read failed.")
            break

        frame = cv2.flip(frame, 1)

        # ── FPS ──
        now       = time.time()
        fps       = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-9))
        prev_time = now

        # ── Pre-processing for face detection ──
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)   # low-light improvement
        faces  = face_cascade.detectMultiScale(
            gray_eq, scaleFactor=1.1,
            minNeighbors=5, minSize=(50, 50))

        # ── Per-face prediction ──
        for (fx, fy, fw, fh) in faces:
            # Crop + resize + normalise
            roi = gray[fy:fy+fh, fx:fx+fw]
            roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            roi_input = roi.astype("float32") / 255.0
            roi_input = roi_input[np.newaxis, ..., np.newaxis]  # (1,64,64,1)

            # Predict
            preds      = model.predict(roi_input, verbose=0)[0]
            top_idx    = int(np.argmax(preds))
            confidence = float(preds[top_idx])
            emotion    = labels.get(top_idx, "unknown")
            color      = EMOTION_COLORS.get(emotion, DEFAULT_COLOR)
            emoji_txt  = EMOTION_EMOJI.get(emotion, "")

            # Bounding box
            cv2.rectangle(frame,
                          (fx, fy), (fx+fw, fy+fh),
                          color, 2)

            # Label background
            label_txt = f"{emotion.upper()}  {confidence*100:.0f}%"
            (lw, lh), _ = cv2.getTextSize(
                label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame,
                          (fx, fy - lh - 12), (fx + lw + 10, fy),
                          color, -1)
            cv2.putText(frame, label_txt,
                        (fx + 5, fy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (10, 10, 10), 2)

            # Confidence bar (below box)
            _draw_bar(frame, fx, fy+fh+2, fw, emotion, confidence, color)

            # Emoji
            cv2.putText(frame, emoji_txt,
                        (fx + fw + 6, fy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                        color, 2)

            # Log with cooldown
            last_log = log_cooldown.get(emotion, 0)
            if now - last_log > LOG_INTERVAL:
                _log_prediction(emotion, confidence)
                log_cooldown[emotion] = now

        # ── HUD ──
        _draw_hud(frame, fps, len(faces))

        cv2.imshow("Use Mode — Facial Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    _status(f"Use Mode closed. Predictions logged → {LOG_FILE}")


# ── CLI entry point ───────────────────────────────
if __name__ == "__main__":
    run_use_mode()
