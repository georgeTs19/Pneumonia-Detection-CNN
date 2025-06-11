import os
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
from src.data_loader import load_data

def plot_metrics(history, output_dir):
    metrics = ["accuracy", "loss"]
    for metric in metrics:
        plt.figure()
        plt.plot(history.history[metric], label=f"train_{metric}")
        plt.plot(history.history[f"val_{metric}"], label=f"val_{metric}")
        plt.title(metric.capitalize())
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{metric}_curve_transfer.png"))
        plt.close()

def build_transfer_model(input_shape):
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=input_shape))
    base_model.trainable = False  # Freeze base model

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train():
    data_dir = "data/chest_xray"
    img_size = (224, 224)
    batch_size = 32
    epochs = 10
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("[INFO] Loading data...")
    train_gen, val_gen, test_gen = load_data(data_dir, img_size=img_size, batch_size=batch_size)

    print("[INFO] Building transfer model...")
    model = build_transfer_model(input_shape=img_size + (3,))
    model.summary()

    checkpoint_path = os.path.join(output_dir, "best_transfer_model.keras")
    checkpoint_cb = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    )

    early_stop_cb = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    print("[INFO] Starting training (transfer learning)...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[checkpoint_cb, early_stop_cb]
    )

    print("[INFO] Training complete. Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"[RESULT] Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    print("[INFO] Saving training history and plots...")
    with open(os.path.join(output_dir, "training_history_transfer.json"), "w") as f:
        json.dump(history.history, f)

    plot_metrics(history, output_dir)

    print("[INFO] Generating classification report...")
    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen)
    y_pred = (y_pred_probs > 0.5).astype("int").flatten()

    print(classification_report(y_true, y_pred, target_names=test_gen.class_indices.keys()))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    train()
