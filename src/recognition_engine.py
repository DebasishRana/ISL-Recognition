"""
Functionality:
-------------
This module provides the core real-time recognition engine.

The RecognitionEngine class:
1. Receives unified multimodal feature vectors.
2. Maintains a rolling 30-frame sequence.
3. Runs the trained GRU model at a controlled interval.
4. Smooths recent predictions.
5. Uses confidence and temporal consistency.
6. Implements charging, commit, cooldown, and release states.
7. Prevents repeated acceptance of a held sign.
8. Supports continuous recognition of different signs.
9. Produces recognition events for the application layer.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class RecognitionEvent:
    sign: str
    confidence: float
    timestamp: float
    frame_count: int
    duration_seconds: float


class RecognitionEngine:

    def __init__(
        self,
        model,
        classes,
        sequence_length=30,
        feature_count=1662,
        inference_interval=3,
        stability_window=5,
        confidence_threshold=0.70,
        release_window=3,
    ):

        self.model = model
        self.classes = list(classes)

        self.sequence_length = (
            sequence_length
        )

        self.feature_count = (
            feature_count
        )

        self.inference_interval = (
            inference_interval
        )

        self.stability_window = (
            stability_window
        )

        self.confidence_threshold = (
            confidence_threshold
        )

        self.release_window = (
            release_window
        )

        self.sequence_buffer = deque(
            maxlen=self.sequence_length
        )

        self.prediction_history = deque(
            maxlen=self.stability_window
        )

        self.confidence_history = deque(
            maxlen=self.stability_window
        )

        self.frame_timestamps = deque(
            maxlen=self.sequence_length
        )

        self.frame_count = 0
        self.last_inference_frame = 0

        self.current_prediction = None
        self.current_confidence = 0.0

        self.stable_prediction = None
        self.stable_confidence = 0.0

        self.last_accepted_sign = None

        self.state = "IDLE"

        self.release_count = 0

        self.sign_start_timestamp = None

        self.last_event = None

    def reset(self):

        self.sequence_buffer.clear()
        self.prediction_history.clear()
        self.confidence_history.clear()
        self.frame_timestamps.clear()

        self.frame_count = 0
        self.last_inference_frame = 0

        self.current_prediction = None
        self.current_confidence = 0.0

        self.stable_prediction = None
        self.stable_confidence = 0.0

        self.last_accepted_sign = None

        self.state = "IDLE"

        self.release_count = 0

        self.sign_start_timestamp = None

        self.last_event = None

    def _validate_features(
        self,
        features,
    ):

        features = np.asarray(
            features,
            dtype=np.float32,
        )

        expected_shape = (
            self.feature_count,
        )

        if features.shape != expected_shape:
            raise ValueError(
                "Unexpected feature shape: "
                f"{features.shape}. "
                f"Expected: {expected_shape}"
            )

        if not np.all(
            np.isfinite(features)
        ):
            raise ValueError(
                "Feature vector contains "
                "NaN or infinite values."
            )

        return features

    def _run_inference(self):

        if (
            len(self.sequence_buffer)
            < self.sequence_length
        ):
            return

        sequence = np.asarray(
            self.sequence_buffer,
            dtype=np.float32,
        )

        sequence = np.expand_dims(
            sequence,
            axis=0,
        )

        probabilities = self.model.predict(
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

        if (
            predicted_index
            >= len(self.classes)
        ):
            raise ValueError(
                "Model predicted an index "
                "outside the class mapping."
            )

        prediction = self.classes[
            predicted_index
        ]

        self.current_prediction = (
            prediction
        )

        self.current_confidence = (
            confidence
        )

        self.last_inference_frame = (
            self.frame_count
        )

        if (
            confidence
            < self.confidence_threshold
        ):

            self._handle_uncertain_prediction()

            return

        self.prediction_history.append(
            prediction
        )

        self.confidence_history.append(
            confidence
        )

        self._update_state()

    def _handle_uncertain_prediction(self):

        self.prediction_history.clear()
        self.confidence_history.clear()

        self.stable_prediction = None
        self.stable_confidence = 0.0

        if self.state == "COMMITTED":

            self.release_count += 1

            if (
                self.release_count
                >= self.release_window
            ):

                self.state = "IDLE"

                self.release_count = 0

                self.last_accepted_sign = None

                self.sign_start_timestamp = None

            return

        if self.state == "CHARGING":

            self.state = "IDLE"

            self.release_count = 0

            self.sign_start_timestamp = None

            return

        self.state = "IDLE"

        self.release_count = 0

    def _update_state(self):

        self.release_count = 0

        if (
            len(self.prediction_history)
            < self.stability_window
        ):

            if self.state == "IDLE":

                self.state = "CHARGING"

            return

        first_prediction = (
            self.prediction_history[0]
        )

        all_same = all(
            prediction == first_prediction
            for prediction
            in self.prediction_history
        )

        if not all_same:

            self.stable_prediction = None
            self.stable_confidence = 0.0

            if self.state == "COMMITTED":

                self.state = "CHARGING"

            else:

                self.state = "IDLE"

            return

        average_confidence = (
            sum(
                self.confidence_history
            )
            / len(
                self.confidence_history
            )
        )

        self.stable_prediction = (
            first_prediction
        )

        self.stable_confidence = (
            average_confidence
        )

        if self.state == "IDLE":

            self.state = "CHARGING"

        if self.state == "CHARGING":

            if (
                self.stable_confidence
                >= self.confidence_threshold
            ):

                if (
                    self.stable_prediction
                    == self.last_accepted_sign
                ):

                    self.state = "COMMITTED"

                    self.prediction_history.clear()
                    self.confidence_history.clear()

                    return

                self.state = "COMMITTED"

                self.sign_start_timestamp = (
                    self.frame_timestamps[0]
                )

                self.prediction_history.clear()
                self.confidence_history.clear()

                self.last_event = (
                    self._create_event()
                )

                self.last_accepted_sign = (
                    self.stable_prediction
                )

    def _create_event(self):

        current_timestamp = (
            self.frame_timestamps[-1]
            if self.frame_timestamps
            else 0.0
        )

        start_timestamp = (
            self.sign_start_timestamp
            if self.sign_start_timestamp
            is not None
            else current_timestamp
        )

        duration = max(
            current_timestamp
            - start_timestamp,
            0.0,
        )

        return RecognitionEvent(
            sign=self.stable_prediction,
            confidence=self.stable_confidence,
            timestamp=current_timestamp,
            frame_count=len(
                self.sequence_buffer
            ),
            duration_seconds=duration,
        )

    def update(
        self,
        features,
        timestamp,
        activity_state="ACTIVE_SIGN",
    ):

        # Convert enum or string
        state_str = (
            activity_state.value
            if hasattr(activity_state, "value")
            else str(activity_state)
        )

        if state_str == "IDLE":
            self.reset()
            return self.get_status()

        features = self._validate_features(
            features
        )

        self.sequence_buffer.append(
            features
        )

        self.frame_timestamps.append(
            float(timestamp)
        )

        self.frame_count += 1

        self.last_event = None

        if state_str == "POSSIBLE_SIGN":
            return {
                "state": "POSSIBLE_SIGN",
                "prediction": "Detecting...",
                "confidence": 0.0,
                "stable_prediction": None,
                "stable_confidence": 0.0,
                "buffer_length": len(self.sequence_buffer),
                "accepted_sign": self.last_accepted_sign,
                "event": None,
            }

        if (
            len(self.sequence_buffer)
            == self.sequence_length
            and (
                self.frame_count
                - self.last_inference_frame
                >= self.inference_interval
            )
        ):

            self._run_inference()

        return self.get_status()

    def get_status(self):

        event = self.last_event

        self.last_event = None

        return {
            "state": self.state,
            "prediction": (
                self.current_prediction
            ),
            "confidence": (
                self.current_confidence
            ),
            "stable_prediction": (
                self.stable_prediction
            ),
            "stable_confidence": (
                self.stable_confidence
            ),
            "buffer_length": (
                len(self.sequence_buffer)
            ),
            "accepted_sign": (
                self.last_accepted_sign
            ),
            "event": event,
        }

    def get_state(self):

        return self.state

    def get_current_prediction(self):

        return self.current_prediction

    def get_current_confidence(self):

        return self.current_confidence

    def get_stable_prediction(self):

        return self.stable_prediction

    def get_buffer_length(self):

        return len(
            self.sequence_buffer
        )