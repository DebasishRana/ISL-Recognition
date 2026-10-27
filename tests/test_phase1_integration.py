"""
Integration tests for Phase 1: Sign / Non-Sign Detection.

Verifies end-to-end integration between:
1. SignActivityDetector
2. RecognitionEngine
3. Gating logic (no predictions, no events, no TTS/logs when IDLE)
"""

import json
import unittest
from pathlib import Path
import numpy as np
import tensorflow as tf

from recognition_engine import RecognitionEngine
from sign_activity_detector import SignActivityState


class TestPhase1Integration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        project_root = Path(__file__).resolve().parents[1]
        model_path = project_root / "models" / "gru_multimodal.keras"
        classes_path = project_root / "models" / "classes.json"

        cls.model = tf.keras.models.load_model(model_path)
        with open(classes_path, "r", encoding="utf-8") as f:
            cls.classes = json.load(f)

    def setUp(self):
        self.engine = RecognitionEngine(
            model=self.model,
            classes=self.classes,
            sequence_length=30,
            feature_count=1662,
            inference_interval=3,
            stability_window=5,
            confidence_threshold=0.70,
        )
        self.dummy_features = np.zeros(1662, dtype=np.float32)

    def test_idle_gating_prevents_inference_and_events(self):
        """When IDLE: buffer remains empty, no inference runs, no predictions/events produced."""
        for i in range(25):
            status = self.engine.update(
                self.dummy_features,
                timestamp=i * 0.033,
                activity_state=SignActivityState.IDLE,
            )
            self.assertEqual(status["state"], "IDLE")
            self.assertIsNone(status["prediction"])
            self.assertEqual(status["confidence"], 0.0)
            self.assertIsNone(status["event"])
            self.assertEqual(status["buffer_length"], 0)

    def test_possible_sign_buffers_without_committing(self):
        """When POSSIBLE_SIGN: buffer accumulates onset frames but inference does not commit events."""
        for i in range(6):
            status = self.engine.update(
                self.dummy_features,
                timestamp=i * 0.033,
                activity_state=SignActivityState.POSSIBLE_SIGN,
            )
            self.assertEqual(status["state"], "POSSIBLE_SIGN")
            self.assertEqual(status["prediction"], "Detecting...")
            self.assertIsNone(status["event"])
            self.assertEqual(status["buffer_length"], i + 1)

    def test_active_sign_executes_inference_and_transitions(self):
        """When ACTIVE_SIGN: sequence buffer fills, inference runs, and state machine advances."""
        # Warmup buffer
        for i in range(30):
            status = self.engine.update(
                self.dummy_features,
                timestamp=i * 0.033,
                activity_state=SignActivityState.ACTIVE_SIGN,
            )
        self.assertEqual(status["buffer_length"], 30)
        self.assertIsNotNone(status["prediction"])

    def test_transition_from_active_to_idle_flushes_pipeline(self):
        """When returning from ACTIVE_SIGN to IDLE: buffer and predictions are flushed immediately."""
        # 1. Fill buffer in ACTIVE_SIGN
        for i in range(30):
            self.engine.update(
                self.dummy_features,
                timestamp=i * 0.033,
                activity_state=SignActivityState.ACTIVE_SIGN,
            )

        # 2. Activity drops to IDLE
        status = self.engine.update(
            self.dummy_features,
            timestamp=30 * 0.033,
            activity_state=SignActivityState.IDLE,
        )
        self.assertEqual(status["state"], "IDLE")
        self.assertEqual(status["buffer_length"], 0)
        self.assertIsNone(status["prediction"])
        self.assertEqual(status["confidence"], 0.0)
        self.assertIsNone(status["event"])


if __name__ == "__main__":
    unittest.main()
