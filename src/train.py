"""
Functionality:
-------------
This module trains the baseline GRU model for Indian Sign
Language recognition.

It loads the prepared landmark sequences, builds a lightweight
two-layer GRU classifier, trains the model with validation
monitoring, evaluates it on the test set, and saves the trained
model, training history, evaluation metrics, classification
report, confusion matrix, and training curves.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


DATASET_ROOT = Path(r"D:\ISL-Dataset")
PROCESSED_ROOT = DATASET_ROOT / "extracted" / "processed_multimodal"
RESULTS_ROOT = PROCESSED_ROOT / "results"
MODEL_ROOT = Path(__file__).resolve().parents[1] / "models"

X_TRAIN_PATH = PROCESSED_ROOT / "X_train.npy"
Y_TRAIN_PATH = PROCESSED_ROOT / "y_train.npy"
X_VAL_PATH = PROCESSED_ROOT / "X_val.npy"
Y_VAL_PATH = PROCESSED_ROOT / "y_val.npy"
X_TEST_PATH = PROCESSED_ROOT / "X_test.npy"
Y_TEST_PATH = PROCESSED_ROOT / "y_test.npy"
CLASSES_PATH = PROCESSED_ROOT / "classes.json"

MODEL_PATH = MODEL_ROOT / "gru_multimodal.keras"

SEED = 42
BATCH_SIZE = 32
EPOCHS = 100


def load_data():
    X_train = np.load(X_TRAIN_PATH)
    y_train = np.load(Y_TRAIN_PATH)

    X_val = np.load(X_VAL_PATH)
    y_val = np.load(Y_VAL_PATH)

    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    with open(
        CLASSES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        classes = json.load(file)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        classes,
    )


def build_model(num_classes):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(30, 1662)
            ),
            tf.keras.layers.GRU(
                128,
                return_sequences=True,
            ),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.GRU(
                64,
            ),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(
                64,
                activation="relu",
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(
                num_classes,
                activation="softmax",
            ),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def save_training_curves(history):
    history_df = pd.DataFrame(history.history)

    history_df.to_csv(
        RESULTS_ROOT / "training_history.csv",
        index=False,
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy",
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("GRU Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        RESULTS_ROOT / "accuracy_curve.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    plt.figure(figsize=(10, 6))

    plt.plot(
        history.history["loss"],
        label="Training Loss",
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GRU Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        RESULTS_ROOT / "loss_curve.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_classification_results(
    y_true,
    y_pred,
    classes,
):
    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(classes)),
        target_names=classes,
        zero_division=0,
        output_dict=True,
    )

    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(
        RESULTS_ROOT / "classification_report.csv"
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(classes)),
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=classes,
        columns=classes,
    )

    matrix_df.to_csv(
        RESULTS_ROOT / "confusion_matrix.csv"
    )

    plt.figure(
        figsize=(18, 16)
    )

    plt.imshow(matrix)

    plt.title("GRU Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")

    plt.xticks(
        range(len(classes)),
        classes,
        rotation=90,
        fontsize=7,
    )

    plt.yticks(
        range(len(classes)),
        classes,
        fontsize=7,
    )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        RESULTS_ROOT / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    metrics = {
        "test_accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "test_samples": int(len(y_true)),
        "num_classes": int(len(classes)),
    }

    with open(
        RESULTS_ROOT / "evaluation_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    print()
    print("TEST RESULTS")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    return metrics


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading training data...")

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        classes,
    ) = load_data()

    print()
    print("DATASET")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Input shape: {X_train.shape[1:]}")
    print(f"Classes: {len(classes)}")

    print()
    print("Building GRU model...")

    model = build_model(
        num_classes=len(classes)
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print()
    print("Starting training...")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val,
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    print()
    print("Loading best model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print()
    print("Evaluating test set...")

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    probabilities = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    y_pred = np.argmax(
        probabilities,
        axis=1,
    )

    save_training_curves(history)

    metrics = save_classification_results(
        y_test,
        y_pred,
        classes,
    )

    metrics["test_loss"] = float(test_loss)

    with open(
        RESULTS_ROOT / "evaluation_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    print()
    print("TRAINING COMPLETE")
    print(f"Best model: {MODEL_PATH}")
    print(f"Results: {RESULTS_ROOT}")


if __name__ == "__main__":
    main()