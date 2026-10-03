"""
Functionality:
-------------
This module provides the real-time ISL recognition application.

The application:
1. Opens the webcam.
2. Tracks hands, pose, and face landmarks.
3. Extracts the unified multimodal feature vector.
4. Sends features to RecognitionEngine.
5. Displays recognition state and predictions.
6. Logs committed signs.
7. Speaks committed signs using Piper.
8. Allows the user to switch between Piper voices.
"""

from pathlib import Path
import json
import time

import cv2
import numpy as np
import tensorflow as tf
import onnxruntime as ort

from multimodal_tracker import MultimodalTracker
from feature_extractor import FeatureExtractor
from recognition_engine import RecognitionEngine
from sign_logger import SignLogger
from tts_engine import TTSEngine


FEATURE_COUNT = 1662

CONFIDENCE_THRESHOLD = 0.70
STABILITY_WINDOW = 5
INFERENCE_INTERVAL = 3

SEQUENCE_LENGTH = 30

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = Path(
    r"D:\ISL-Dataset"
)

PROCESSED_ROOT = (
    DATASET_ROOT
    / "extracted"
    / "processed_multimodal"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "gru_multimodal.onnx"
)

CLASSES_PATH = (
    PROJECT_ROOT
    / "models"
    / "classes.json"
)

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    input_shape = (None, SEQUENCE_LENGTH, FEATURE_COUNT)
    output_shape = (
        None,
        session.get_outputs()[0].shape[-1],
    )

    class ONNXModelAdapter:
        def __init__(self):
            self.session = session
            self.input_name = input_name
            self.input_shape = input_shape
            self.output_shape = output_shape

        def predict(self, sequence, verbose=0):
            sequence = np.asarray(
                sequence,
                dtype=np.float32,
            )

            return self.session.run(
                None,
                {self.input_name: sequence},
            )[0]

    return ONNXModelAdapter()


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
            "classes.json must contain a list."
        )

    if not classes:
        raise ValueError(
            "classes.json contains no classes."
        )

    return classes


def validate_model(
    model,
    classes,
):

    output_classes = (
        model.output_shape[-1]
    )

    if output_classes != len(classes):
        raise ValueError(
            "Model output classes do not match "
            f"class mapping. Model: {output_classes}, "
            f"Classes: {len(classes)}"
        )

    expected_shape = (
        None,
        SEQUENCE_LENGTH,
        FEATURE_COUNT,
    )

    if tuple(model.input_shape) != expected_shape:
        raise ValueError(
            "Unexpected model input shape. "
            f"Model: {model.input_shape}, "
            f"Expected: {expected_shape}"
        )


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

    engine = RecognitionEngine(
        model=model,
        classes=classes,
        sequence_length=SEQUENCE_LENGTH,
        feature_count=FEATURE_COUNT,
        inference_interval=INFERENCE_INTERVAL,
        stability_window=STABILITY_WINDOW,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        release_window=3,
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        tracker.close()
        tts.close()

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
    state = "IDLE"

    voices = tts.get_voices()
    voice_index = 0

    if voices:

        tts.set_voice(
            voices[voice_index]
        )

    try:

        while True:

            success, frame = camera.read()

            if not success:

                print(
                    "Failed to read frame "
                    "from webcam."
                )

                break

#flip image
            # frame = cv2.flip(
            #     frame,
            #     1,
            # )

            timestamp = time.time()

            timestamp_ms = int(
                timestamp * 1000
            )

            tracking_result = (
                tracker.process(
                    frame,
                    timestamp_ms,
                )
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

            status = engine.update(
                features,
                timestamp,
            )

            state = status[
                "state"
            ]

            predicted_label = (
                status[
                    "prediction"
                ]
                or "Waiting..."
            )

            confidence = float(
                status[
                    "confidence"
                ]
            )

            stable_label = (
                status[
                    "stable_prediction"
                ]
                or "None"
            )

            event = status[
                "event"
            ]

            if event is not None:

                print(
                    f"Accepted sign: "
                    f"{event.sign} "
                    f"({event.confidence * 100:.1f}%)"
                )

                logger.log_sign(
                    sign=event.sign,
                    confidence=event.confidence,
                    duration_seconds=(
                        event.duration_seconds
                    ),
                    frame_count=(
                        event.frame_count
                    ),
                    status="ACCEPTED",
                )

                tts.speak(
                    event.sign
                )

            current_time = time.time()

            fps = 1.0 / max(
                current_time
                - previous_time,
                1e-6,
            )

            previous_time = (
                current_time
            )

            current_voice = (
                tts.get_current_voice()
                if voices
                else "None"
            )

            cv2.putText(
                frame,
                f"Prediction: "
                f"{predicted_label}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Confidence: "
                f"{confidence * 100:.1f}%",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"State: {state}",
                (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Stable: {stable_label}",
                (30, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Buffer: "
                f"{engine.get_buffer_length()}/"
                f"{SEQUENCE_LENGTH}",
                (30, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Classes: {len(classes)}",
                (30, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (30, 290),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Voice: {current_voice}",
                (30, 330),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                "1-9: Voice | Q: Quit",
                (30, 370),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Real-Time ISL Recognition",
                frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

            if (
                ord("1")
                <= key
                <= ord("9")
            ):

                selected_index = (
                    key - ord("1")
                )

                if (
                    selected_index
                    < len(voices)
                ):

                    voice_index = (
                        selected_index
                    )

                    tts.set_voice(
                        voices[
                            voice_index
                        ]
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