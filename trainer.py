import os
import time
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "8"

DATASET_DIR = "dataset"
MODEL_PATH  = "model.keras"
IMG_SIZE    = 48
EMOTIONS    = ["angry", "fear", "happy", "neutral", "sad", "surprise"]

def _load_dataset(status_callback=None):
    def _log(msg):
        if status_callback: status_callback(msg)
        print(msg)
    X, y = [], []
    for label_idx, emotion in enumerate(EMOTIONS):
        folder = os.path.join(DATASET_DIR, emotion)
        if not os.path.isdir(folder): continue
        files = [f for f in os.listdir(folder) if f.lower().endswith((".png",".jpg",".jpeg"))]
        _log(f"  Loading {emotion}: {len(files)} images")
        for fname in files:
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            X.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
            y.append(label_idx)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def _build_cnn(num_classes):
    from tensorflow.keras import layers, models
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs=inp, outputs=out, name="EmotionCNN_v4")

def _save_plot(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
    axes[0].plot(history.history["accuracy"], color="#58a6ff", label="Train", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], color="#3fb950", label="Val", linewidth=2)
    axes[0].set_title("Accuracy", color="#e6edf3")
    axes[0].legend(facecolor="#21262d", labelcolor="#e6edf3")
    axes[1].plot(history.history["loss"], color="#f85149", label="Train", linewidth=2)
    axes[1].plot(history.history["val_loss"], color="#d29922", label="Val", linewidth=2)
    axes[1].set_title("Loss", color="#e6edf3")
    axes[1].legend(facecolor="#21262d", labelcolor="#e6edf3")
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=120, facecolor="#0d1117")
    plt.close()
    print("Plot saved!")

def train_model(status_callback=None, epochs=50):
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.utils import to_categorical

    def _log(msg):
        if status_callback: status_callback(msg)
        print(msg)

    _log("Loading dataset...")
    X, y = _load_dataset(status_callback)
    if len(X) < 30:
        _log("Not enough data!")
        return

    num_classes = len(np.unique(y))
    _log(f"Dataset: {len(X)} images, {num_classes} classes.")
    X = X / 255.0
    X = X[..., np.newaxis]
    y_cat = to_categorical(y, num_classes)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.20, random_state=42, stratify=y)
    _log(f"Train: {len(X_train)}  Val: {len(X_val)}")

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y), y=y)
    class_weight_dict = dict(enumerate(class_weights))
    _log(f"Class weights: {class_weight_dict}")

    aug = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
    aug.fit(X_train)

    _log("Building CNN v4...")
    model = _build_cnn(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"])
    model.summary(print_fn=_log)

    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1),
    ]

    _log(f"Training for up to {epochs} epochs...")
    t0 = time.time()
    history = model.fit(
        aug.flow(X_train, y_train, batch_size=128),
        validation_data=(X_val, y_val),
        epochs=epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1)

    elapsed = time.time() - t0
    best_val_acc = max(history.history["val_accuracy"])
    _log(f"Done in {elapsed:.1f}s | Best val accuracy: {best_val_acc:.2%}")

    with open("label_map.json", "w") as f:
        json.dump({i: e for i, e in enumerate(EMOTIONS)}, f)
    _log("label_map.json saved!")
    _save_plot(history)

if __name__ == "__main__":
    train_model()
