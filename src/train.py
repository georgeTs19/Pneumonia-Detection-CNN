import os
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
from src.data_loader import load_data  # type: ignore
from src.model import build_custom_cnn  # type: ignore

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
        plt.savefig(os.path.join(output_dir, f"{metric}_curve.png"))
        plt.close()

def train():
    # Training configuration
    data_dir = "data/chest_xray"
    img_size = (224, 224)
    batch_size = 32
    epochs = 10
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Load the data
    print("[INFO] Loading data...")
    train_gen, val_gen, test_gen = load_data(data_dir, img_size=img_size, batch_size=batch_size)

    # Build the model
    print("[INFO] Building model...")
    model = build_custom_cnn(input_shape=img_size + (3,))# (224, 224, 3)
    print("[INFO] Model summary:")
    model.summary()


    # Setup callbacks
    checkpoint_path = os.path.join(output_dir, "best_model.keras")
    checkpoint_cb = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    )

    early_stop_cb = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    steps_per_epoch = train_gen.samples // train_gen.batch_size
    validation_steps = val_gen.samples // val_gen.batch_size

    # Train the model
    print("[INFO] Starting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=[checkpoint_cb, early_stop_cb]    
    )


    print("[INFO] Training complete.")
    print(f"[INFO] Best model saved to: {checkpoint_path}")

    # Save training history
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history.history, f)
    print("[INFO] Training history saved.")

    # Plot training curves
    plot_metrics(history, output_dir)
    print("[INFO] Accuracy and loss curves saved.")

    # Evaluate on test set
    print("[INFO] Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"[RESULT] Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    # Classification report & confusion matrix
    print("[INFO] Generating classification report...")
    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen)
    y_pred = (y_pred_probs > 0.5).astype("int").flatten()

    print(classification_report(y_true, y_pred, target_names=test_gen.class_indices.keys()))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    with open(os.path.join(output_dir, "test_results.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=test_gen.class_indices.keys()))
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_true, y_pred)))

    print("[INFO] Test results saved to test_results.txt.")


    return history

# Ensures this script only runs when executed directly
if __name__ == "__main__":
    train()
