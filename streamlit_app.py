import streamlit as st
import os, json, csv, time
import numpy as np
from datetime import datetime
from pathlib import Path

# ── Page config ──────────────────────────────────
st.set_page_config(
    page_title="Facial Emotion Detection",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────
st.markdown("""
<style>
  body { background-color: #0d1117; }
  .stApp { background-color: #0d1117; color: #e6edf3; }
  .metric-card {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 16px;
      text-align: center;
  }
  .emotion-badge {
      display: inline-block;
      background: #21262d;
      border-radius: 20px;
      padding: 4px 12px;
      margin: 4px;
      font-size: 13px;
  }
</style>
""", unsafe_allow_html=True)

EMOTIONS   = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
DATASET_DIR = "dataset"
LOG_FILE    = "emotion_log.csv"
MODEL_PATH  = "model.hdf5"


# ── Sidebar ───────────────────────────────────────
st.sidebar.title("🎭 Emotion Detection")
st.sidebar.markdown("---")
mode = st.sidebar.radio("Select Mode", ["📊 Dashboard", "🧠 Train", "📷 Use Mode"])


# ──────────────────────────────────────────────────
#  DASHBOARD
# ──────────────────────────────────────────────────
if "Dashboard" in mode:
    st.title("📊 System Dashboard")

    col1, col2, col3 = st.columns(3)

    # Dataset stats
    total_imgs = 0
    counts = {}
    for emo in EMOTIONS:
        folder = os.path.join(DATASET_DIR, emo)
        n = len(list(Path(folder).glob("*.png"))) if Path(folder).exists() else 0
        counts[emo] = n
        total_imgs += n

    with col1:
        st.metric("Total Training Images", total_imgs)
    with col2:
        model_exists = os.path.exists(MODEL_PATH)
        st.metric("Model", "✅ Ready" if model_exists else "❌ Not trained")
    with col3:
        log_rows = 0
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE) as f:
                log_rows = sum(1 for _ in f) - 1
        st.metric("Predictions Logged", log_rows)

    st.markdown("---")
    st.subheader("Dataset per Emotion")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    bars = ax.bar(counts.keys(), counts.values(),
                  color=["#f85149","#d29922","#3fb950",
                         "#8b949e","#58a6ff","#bc8cff"])
    ax.tick_params(colors="#e6edf3")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.set_ylabel("Images", color="#e6edf3")
    for bar, val in zip(bars, counts.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', color="#e6edf3", fontsize=10)
    st.pyplot(fig)

    # Training plot
    if os.path.exists("training_history.png"):
        st.markdown("---")
        st.subheader("Training History")
        st.image("training_history.png")

    # Recent log
    if os.path.exists(LOG_FILE):
        st.markdown("---")
        st.subheader("Recent Predictions")
        import pandas as pd
        df = pd.read_csv(LOG_FILE)
        st.dataframe(df.tail(20), use_container_width=True)


# ──────────────────────────────────────────────────
#  TRAIN
# ──────────────────────────────────────────────────
elif "Train" in mode:
    st.title("🧠 Train Model")
    st.info(
        "**Step 1**: Collect face images via the desktop app (Train Mode).\n\n"
        "**Step 2**: Click **Train Model** below to train the CNN.",
        icon="ℹ️")

    use_mobilenet = st.checkbox("Use MobileNetV2 (transfer learning — slower but more accurate)")
    epochs        = st.slider("Epochs", 5, 60, 25)

    if st.button("🚀 Train Model", type="primary"):
        from trainer import train_model
        log_output = st.empty()
        messages   = []

        def cb(msg):
            messages.append(msg)
            log_output.code("\n".join(messages[-20:]))

        with st.spinner("Training in progress…"):
            train_model(use_mobilenet=use_mobilenet,
                        status_callback=cb,
                        epochs=epochs)

        st.success("✅ Training complete! Model saved as model.hdf5")
        if os.path.exists("training_history.png"):
            st.image("training_history.png")


# ──────────────────────────────────────────────────
#  USE MODE (upload image)
# ──────────────────────────────────────────────────
elif "Use Mode" in mode:
    st.title("📷 Emotion Detection — Upload Image")
    st.info("Upload a photo to detect emotions. "
            "For live webcam, use the desktop app.", icon="📷")

    if not os.path.exists(MODEL_PATH):
        st.error("No trained model found. Please train first!")
        st.stop()

    uploaded = st.file_uploader("Upload a face image",
                                type=["jpg", "jpeg", "png"])
    if uploaded:
        import cv2
        import tensorflow as tf
        from PIL import Image
        import io

        img_bytes = uploaded.read()
        nparr     = np.frombuffer(img_bytes, np.uint8)
        img_bgr   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Load model + labels
        model  = tf.keras.models.load_model(MODEL_PATH)
        labels = {}
        if os.path.exists("label_map.json"):
            with open("label_map.json") as f:
                labels = {int(k): v for k, v in json.load(f).items()}
        else:
            labels = {i: e for i, e in enumerate(EMOTIONS)}

        # Detect faces
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces   = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))

        col1, col2 = st.columns([2, 1])

        with col1:
            if len(faces) == 0:
                st.warning("No faces detected in the image.")
                st.image(img_rgb, use_column_width=True)
            else:
                annotated = img_bgr.copy()
                results   = []
                for (fx, fy, fw, fh) in faces:
                    roi       = gray[fy:fy+fh, fx:fx+fw]
                    roi       = cv2.resize(roi, (64, 64))
                    roi_input = roi.astype("float32") / 255.0
                    roi_input = roi_input[np.newaxis, ..., np.newaxis]
                    preds     = model.predict(roi_input, verbose=0)[0]
                    top_idx   = int(np.argmax(preds))
                    conf      = float(preds[top_idx])
                    emotion   = labels.get(top_idx, "unknown")
                    results.append((emotion, conf, preds))
                    cv2.rectangle(annotated, (fx, fy), (fx+fw, fy+fh),
                                  (0, 230, 100), 2)
                    cv2.putText(annotated,
                                f"{emotion} {conf*100:.0f}%",
                                (fx, fy - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (0, 230, 100), 2)

                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                         use_column_width=True)

        with col2:
            if len(faces) > 0:
                for i, (emotion, conf, preds) in enumerate(results):
                    st.markdown(f"**Face {i+1}**: `{emotion.upper()}`")
                    st.progress(conf)
                    st.caption(f"Confidence: {conf*100:.1f}%")
                    st.markdown("---")
                    # Full distribution
                    for idx, prob in enumerate(preds):
                        lbl = labels.get(idx, str(idx))
                        st.caption(f"{lbl}: {prob*100:.1f}%")
                        st.progress(float(prob))
