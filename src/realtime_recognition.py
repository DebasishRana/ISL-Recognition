"""
Functionality: This module provides real-time Indian Sign Language recognition.

"""

from collections import deque
from pathlib import Path
import json
import time

import cv2
import numpy as np
import tensorflow as tf

from multimodal_tracker import MultimodalTracker
from feature_extractor import FeatureExtractor
from sign_logger import SignLogger
from tts_engine import TTSEngine


SEQUENCE_LENGTH = 30
FEATURE_COUNT = 1662

CONFIDENCE_THRESHOLD = 0.70
STABILITY_WINDOW = 5
INFERENCE_INTERVAL = 3

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = Path(r"D:\ISL-Dataset")

PROCESSED_ROOT = (
    DATASET_ROOT
    / "extracted"
    / "processed_multimodal"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "gru_multimodal.keras"
)

CLASSES_PATH = (
    PROCESSED_ROOT
    / "classes.json"
)


def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH
    )


def load_classes():

    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Class mapping not found at: {CLASSES_PATH}"
        )

    with open(
        CLASSES_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        classes = json.load(file)

    if not isinstance(classes, list):
        raise ValueError(
            "classes.json must contain a list of class names."
        )

    if not classes:
        raise ValueError(
            "classes.json contains no classes."
        )

    return classes


def validate_model(model, classes):

    output_classes = model.output_shape[-1]

    if output_classes != len(classes):
        raise ValueError(
            "Model output classes do not match "
            f"class mapping. Model: {output_classes}, "
            f"Classes: {len(classes)}"
        )


def select_voice(tts, voice_index):

    voices = tts.get_voices()

    if not voices:
        return voice_index

    if 0 <= voice_index < len(voices):
        tts.set_voice(
            voices[voice_index]
        )

    return voice_index


def main():

    model = load_model()
    classes = load_classes()

    validate_model(
        model,
        classes,
    )

    tracker = MultimodalTracker()
    extractor = FeatureExtractor()
    logger = SignLogger()
    tts = TTSEngine()

    sequence_buffer = deque(
        maxlen=SEQUENCE_LENGTH
    )

    timestamp_buffer = deque(
        maxlen=SEQUENCE_LENGTH
    )

    prediction_history = deque(
        maxlen=STABILITY_WINDOW
    )

    confidence_history = deque(
        maxlen=STABILITY_WINDOW
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        tracker.close()

        raise RuntimeError(
            "Could not open webcam."
        )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720,
    )

    previous_time = time.time()

    predicted_label = "Waiting..."
    confidence = 0.0
    stable_label = "Waiting..."
    accepted_label = None
    last_inference_frame = 0
    frame_count = 0
    voice_index = 0

    voices = tts.get_voices()

    if voices:
        tts.set_voice(
            voices[voice_index]
        )

    try:

        while True:

            success, frame = camera.read()

            if not success:

                print(
                    "Failed to read frame from webcam."
                )

                break

            frame = cv2.flip(
                frame,
                1,
            )

            timestamp_ms = int(
                time.time() * 1000
            )

            tracking_result = tracker.process(
                frame,
                timestamp_ms,
            )

            features = extractor.extract(
                tracking_result
            )

            features = np.asarray(
                features,
                dtype=np.float32,
            )

            if features.shape != (
                FEATURE_COUNT,
            ):

                raise ValueError(
                    "Unexpected feature shape: "
                    f"{features.shape}. "
                    f"Expected: "
                    f"({FEATURE_COUNT},)"
                )

            sequence_buffer.append(
                features
            )

            timestamp_buffer.append(
                time.time()
            )

            frame_count += 1

            if (
                len(sequence_buffer)
                == SEQUENCE_LENGTH
                and frame_count - last_inference_frame
                >= INFERENCE_INTERVAL
            ):

                sequence = np.asarray(
                    sequence_buffer,
                    dtype=np.float32,
                )

                sequence = np.expand_dims(
                    sequence,
                    axis=0,
                )

                probabilities = model.predict(
                    sequence,
                    verbose=0,
                )[0]

                predicted_index = int(
                    np.argmax(probabilities)
                )

                confidence = float(
                    probabilities[
                        predicted_index
                    ]
                )

                if confidence >= CONFIDENCE_THRESHOLD:

                    predicted_label = classes[
                        predicted_index
                    ]

                    prediction_history.append(
                        predicted_label
                    )

                    confidence_history.append(
                        confidence
                    )

                else:

                    predicted_label = "Uncertain"

                    prediction_history.clear()
                    confidence_history.clear()

                if (
                    len(prediction_history)
                    == STABILITY_WINDOW
                ):

                    first_label = (
                        prediction_history[0]
                    )

                    all_same = all(
                        label == first_label
                        for label in prediction_history
                    )

                    average_confidence = (
                        sum(confidence_history)
                        / len(confidence_history)
                    )

                    if (
                        all_same
                        and average_confidence
                        >= CONFIDENCE_THRESHOLD
                    ):

                        stable_label = first_label

                        if (
                            stable_label
                            != accepted_label
                        ):

                            duration_seconds = 0.0

                            if len(timestamp_buffer) >= 2:

                                duration_seconds = (
                                    timestamp_buffer[-1]
                                    - timestamp_buffer[0]
                                )

                            frame_count_for_log = (
                                len(sequence_buffer)
                            )

                            logger.log_sign(
                                stable_label,
                                average_confidence,
                                duration_seconds,
                                frame_count_for_log,
                                status="ACCEPTED",
                            )

                            tts.speak(
                                stable_label
                            )

                            accepted_label = (
                                stable_label
                            )

                            prediction_history.clear()
                            confidence_history.clear()

                last_inference_frame = (
                    frame_count
                )

            if (
                accepted_label is not None
                and predicted_label != accepted_label
                and predicted_label != "Uncertain"
            ):

                accepted_label = None

            current_time = time.time()

            fps = 1.0 / max(
                current_time - previous_time,
                1e-6,
            )

            previous_time = current_time

            current_voice = (
                tts.get_current_voice()
                if voices
                else "None"
            )

            cv2.putText(
                frame,
                f"Sign: {predicted_label}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence * 100:.1f}%",
                (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Stable: {stable_label}",
                (30, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Voice: {current_voice}",
                (30, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Buffer: {len(sequence_buffer)}/{SEQUENCE_LENGTH}",
                (30, 215),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Classes: {len(classes)}",
                (30, 255),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (30, 295),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                "1-9: Voice | Q: Quit",
                (30, 335),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Real-Time ISL Recognition",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if ord("1") <= key <= ord("9"):

                selected_index = (
                    key - ord("1")
                )

                if selected_index < len(voices):

                    voice_index = selected_index

                    tts.set_voice(
                        voices[voice_index]
                    )

                    print(
                        "Selected voice:",
                        tts.get_current_voice(),
                    )

    finally:

        camera.release()
        cv2.destroyAllWindows()
        tracker.close()
        tts.close()


if __name__ == "__main__":
    main()