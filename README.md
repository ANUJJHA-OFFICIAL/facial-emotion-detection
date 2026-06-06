# 🎭 Facial Emotion Detection System

Real-time facial emotion detection using OpenCV + TensorFlow/Keras with a
Tkinter GUI.  Supports 6 emotions: **Happy, Sad, Angry, Neutral, Surprise, Fear**.

---

## 📁 Project Structure

```
facial_emotion_detection/
│
├── main.py              ← Launch this (Tkinter GUI)
├── train_mode.py        ← Webcam data collection
├── trainer.py           ← CNN / MobileNetV2 training pipeline
├── use_mode.py          ← Real-time prediction
├── streamlit_app.py     ← Optional web UI (streamlit)
│
├── requirements.txt
├── README.md
│
├── dataset/             ← Created automatically
│   ├── happy/
│   ├── sad/
│   ├── angry/
│   ├── neutral/
│   ├── surprise/
│   └── fear/
│
├── model.hdf5           ← Saved after training
├── label_map.json       ← Label index → emotion name
├── training_history.png ← Accuracy / loss plot
└── emotion_log.csv      ← Prediction log (Use Mode)
```

---

## ⚙️ Installation

### 1. Clone / download project
```bash
cd facial_emotion_detection
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **CPU-only machine?**  Replace `tensorflow` with `tensorflow-cpu` in requirements.txt

> **Linux missing Tkinter?**
> ```bash
> sudo apt-get install python3-tk
> ```

---

## 🚀 Running the Application

```bash
python main.py
```

A GUI window opens with three buttons:

| Button | Action |
|--------|--------|
| 🧠 Train Mode | Open webcam to collect face images |
| 📷 Use Mode  | Run real-time emotion detection |
| 🔄 Retrain   | Retrain directly from GUI |
| ✖ Exit       | Close application |

---

## 🎯 Train Mode — Collecting Data

1. Click **Train Mode** — webcam opens
2. Face the camera, then press:

| Key | Emotion  |
|-----|----------|
| H   | Happy    |
| S   | Sad      |
| A   | Angry    |
| N   | Neutral  |
| U   | Surprise |
| F   | Fear     |
| T   | Train Model |
| Q   | Quit     |

3. Press a key → face is **automatically captured & saved** to `dataset/<emotion>/`
4. Repeat until you have **at least 50–100 images per emotion** for decent accuracy
5. Press **T** to start training immediately, or close and use the GUI

---

## 🧠 Training the Model

Training starts when you press **T** inside Train Mode, or click **Retrain Model** in the GUI.

What happens:
- All images in `dataset/` are loaded
- 80/20 train/validation split
- Data augmentation (rotation, zoom, flip)
- Custom **CNN** trains (or **MobileNetV2** with `--mobilenet` flag)
- Best model saved to `model.hdf5`
- Training plot saved to `training_history.png`

CLI training:
```bash
python trainer.py                # custom CNN
python trainer.py --mobilenet    # MobileNetV2 transfer learning
```

---

## 📷 Use Mode — Real-time Detection

1. Make sure `model.hdf5` exists (train first)
2. Click **Use Mode** — webcam opens
3. Emotions, confidence %, and bounding boxes are drawn live
4. All predictions are logged to `emotion_log.csv` with timestamps
5. Press **Q** to quit

---

## 🌐 Streamlit Web UI (optional)

```bash
pip install streamlit pandas
streamlit run streamlit_app.py
```

Features:
- Dashboard with dataset stats and training history
- One-click model training
- Upload a photo to detect emotions

---

## 💡 Tips for Better Accuracy

| Tip | Detail |
|-----|--------|
| **More data** | 200+ images per emotion gives much better results |
| **Varied lighting** | Collect in different lighting conditions |
| **Varied angles** | Slight head tilts, turned faces |
| **Use MobileNetV2** | Enable with `--mobilenet` flag for higher accuracy |
| **More epochs** | Increase epochs to 50–100 if val accuracy still rising |
| **Balanced classes** | Try to have similar counts per emotion |
| **Clean dataset** | Remove blurry or incorrectly-labelled images |

---

## 🐛 Common Issues

| Issue | Solution |
|-------|---------|
| `No module named 'cv2'` | `pip install opencv-python` |
| `No module named 'tensorflow'` | `pip install tensorflow` |
| Webcam not opening | Check camera permissions; try `cv2.VideoCapture(1)` |
| Low accuracy | Collect more data; enable MobileNetV2 |
| Tkinter not found (Linux) | `sudo apt-get install python3-tk` |
| CUDA errors | Install `tensorflow-cpu` instead |

---

## 📊 Expected Accuracy

| Dataset size per emotion | Approx. Val Accuracy |
|--------------------------|----------------------|
| 30–50                   | ~50–60%              |
| 100–200                 | ~65–75%              |
| 500+                    | ~80–85%              |
| FER2013 full dataset    | ~65–70% (industry standard) |

---

## 📝 Emotion Log Format

`emotion_log.csv`:
```
timestamp,emotion,confidence_%
2024-01-15 14:32:10,happy,87.3
2024-01-15 14:32:12,neutral,64.1
```

---

## 🔬 Advanced: Export Model

```python
import tensorflow as tf
model = tf.keras.models.load_model("model.hdf5")

# Export as TFLite (mobile-friendly)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
```
