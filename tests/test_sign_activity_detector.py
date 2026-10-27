"""
Unit tests for SignActivityDetector.
Phase 1: Sign / Non-Sign Detection.

Verifies:
  1. Sitting still
  2. Hands not visible
  3. Hand held still
  4. Small random movement (micro-jitter, single-frame twitch)
  5. Intentional sign movement (smooth gesture trajectory)
"""

import unittest
import numpy as np

from sign_activity_detector import (
    SignActivityDetector,
    SignActivityConfig,
    SignActivityState,
)


class MockLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class MockCategory:
    def __init__(self, category_name: str = "Right"):
        self.category_name = category_name


class MockHandsResult:
    def __init__(self, hand_landmarks=None, handedness=None):
        self.hand_landmarks = hand_landmarks or []
        self.handedness = handedness or []


class MockPoseResult:
    def __init__(self, pose_landmarks=None):
        self.pose_landmarks = pose_landmarks or []


def make_hand(wrist_x: float, wrist_y: float, wrist_z: float = 0.0, jitter: float = 0.0):
    """Generates 21 hand landmarks around a wrist position."""
    landmarks = []
    rng = np.random.RandomState(42) if jitter > 0 else None
    for i in range(21):
        noise = (rng.uniform(-jitter, jitter) if rng else 0.0)
        # Offset fingers slightly from wrist
        offset_y = -0.01 * (i // 4)
        offset_x = 0.005 * (i % 4)
        landmarks.append(MockLandmark(wrist_x + offset_x + noise, wrist_y + offset_y + noise, wrist_z))
    return landmarks


def make_pose(shoulder_span: float = 0.30):
    """Generates a standard seated pose with left and right shoulders."""
    pose = [MockLandmark(0.5, 0.2)] * 33  # filler
    pose[11] = MockLandmark(0.5 - shoulder_span / 2.0, 0.35)  # Left shoulder
    pose[12] = MockLandmark(0.5 + shoulder_span / 2.0, 0.35)  # Right shoulder
    return [pose]


class TestSignActivityDetector(unittest.TestCase):

    def setUp(self):
        self.config = SignActivityConfig(
            motion_start_threshold=0.90,
            motion_active_threshold=1.40,
            motion_continue_threshold=0.60,
            min_possible_frames=2,
            min_active_frames=3,
            cooldown_frames=6,
            possible_timeout_frames=4,
            resting_y_threshold=0.82,
        )
        self.detector = SignActivityDetector(self.config)

    def test_scenario_1_sitting_still(self):
        """1. Sitting still: Hands on desk/lap with sensor jitter -> Must remain IDLE."""
        pose = make_pose()
        # Hands resting at y=0.85 (lap/desk) with minor sensor jitter
        for frame_idx in range(30):
            t = frame_idx * 0.033
            # Micro-jitter of 0.0005 (sensor noise)
            hand = make_hand(0.5, 0.85, jitter=0.0005)
            tracking = {
                "hands": MockHandsResult([hand], [[MockCategory("Right")]]),
                "pose": MockPoseResult(pose),
            }
            status = self.detector.process(tracking, t)
            self.assertEqual(
                status["state"],
                SignActivityState.IDLE,
                f"Failed at frame {frame_idx}: state was {status['state']}"
            )
            self.assertFalse(status["is_signing"])

    def test_scenario_2_hands_not_visible(self):
        """2. Hands not visible: 0 hands detected -> Must remain IDLE."""
        pose = make_pose()
        for frame_idx in range(20):
            t = frame_idx * 0.033
            tracking = {
                "hands": MockHandsResult([], []),
                "pose": MockPoseResult(pose),
            }
            status = self.detector.process(tracking, t)
            self.assertEqual(status["state"], SignActivityState.IDLE)
            self.assertFalse(status["is_signing"])
            self.assertEqual(status["hands_detected"], 0)

    def test_scenario_3_hand_held_still(self):
        """3. Hand held still: Hand raised in signing space (y=0.45) but stationary -> Must remain IDLE."""
        pose = make_pose()
        for frame_idx in range(40):
            t = frame_idx * 0.033
            # Stationary hand in front of chest with minor tracking noise
            hand = make_hand(0.55, 0.45, jitter=0.0008)
            tracking = {
                "hands": MockHandsResult([hand], [[MockCategory("Right")]]),
                "pose": MockPoseResult(pose),
            }
            status = self.detector.process(tracking, t)
            self.assertEqual(
                status["state"],
                SignActivityState.IDLE,
                f"Holding hand still incorrectly triggered {status['state']} at frame {frame_idx}"
            )
            self.assertFalse(status["is_signing"])

    def test_scenario_4_small_random_movement(self):
        """4. Small random movement: Brief 1-frame twitch or tremor -> Must NOT trigger ACTIVE_SIGN."""
        pose = make_pose()
        # 10 stationary frames
        for frame_idx in range(10):
            t = frame_idx * 0.033
            hand = make_hand(0.55, 0.45, jitter=0.0005)
            tracking = {
                "hands": MockHandsResult([hand], [[MockCategory("Right")]]),
                "pose": MockPoseResult(pose),
            }
            status = self.detector.process(tracking, t)
            self.assertEqual(status["state"], SignActivityState.IDLE)

        # 1-frame sudden jerk/twitch
        t = 10 * 0.033
        twitch_hand = make_hand(0.58, 0.48)  # small jerk
        tracking = {
            "hands": MockHandsResult([twitch_hand], [[MockCategory("Right")]]),
            "pose": MockPoseResult(pose),
        }
        status = self.detector.process(tracking, t)
        # Should NOT be ACTIVE_SIGN
        self.assertNotEqual(status["state"], SignActivityState.ACTIVE_SIGN)

        # Hand immediately returns to rest
        for frame_idx in range(11, 20):
            t = frame_idx * 0.033
            hand = make_hand(0.58, 0.48, jitter=0.0005)
            tracking = {
                "hands": MockHandsResult([hand], [[MockCategory("Right")]]),
                "pose": MockPoseResult(pose),
            }
            status = self.detector.process(tracking, t)
            self.assertNotEqual(status["state"], SignActivityState.ACTIVE_SIGN)

        # Should revert to IDLE
        self.assertEqual(status["state"], SignActivityState.IDLE)

    def test_scenario_5_intentional_sign_movement(self):
        """5. Intentional sign movement: Sustained, deliberate gesture trajectory -> IDLE -> POSSIBLE -> ACTIVE -> IDLE."""
        pose = make_pose()
        states_seen = []

        # Part A: Initial rest (5 frames)
        for frame_idx in range(5):
            t = frame_idx * 0.033
            hand = make_hand(0.5, 0.5)
            tracking = {
                "hands": MockHandsResult([hand], [[MockCategory("Right")]]),
                "pose": MockPoseResult(pose),
            }
            status = self.detector.process(tracking, t)
            states_seen.append(status["state"])

        self.assertEqual(states_seen[-1], SignActivityState.IDLE)

        # Part B: Intentional moving gesture across 15 frames (e.g., circular sign motion)
        for frame_idx in range(5, 20):
            t = frame_idx * 0.033
            angle = (frame_idx - 5) * 0.4
            # Moderate velocity trajectory (approx 0.02 - 0.04 screen displacement per frame)
            x = 0.5 + 0.12 * np.cos(angle)
            y = 0.45 + 0.10 * np.sin(angle)
            hand = make_hand(x, y)
            tracking = {
                "hands": MockHandsResult([hand], [[MockCategory("Right")]]),
                "pose": MockPoseResult(pose),
            }
            status = self.detector.process(tracking, t)
            states_seen.append(status["state"])

        # Must have transitioned through POSSIBLE_SIGN to ACTIVE_SIGN
        self.assertIn(SignActivityState.POSSIBLE_SIGN, states_seen)
        self.assertIn(SignActivityState.ACTIVE_SIGN, states_seen)
        self.assertEqual(status["state"], SignActivityState.ACTIVE_SIGN)
        self.assertTrue(status["is_signing"])

        # Part C: Sign finishes, hands stop and hold still
        for frame_idx in range(20, 35):
            t = frame_idx * 0.033
            hand = make_hand(0.5, 0.45, jitter=0.0005)
            tracking = {
                "hands": MockHandsResult([hand], [[MockCategory("Right")]]),
                "pose": MockPoseResult(pose),
            }
            status = self.detector.process(tracking, t)
            states_seen.append(status["state"])

        # After cooldown frames, must safely return to IDLE
        self.assertEqual(status["state"], SignActivityState.IDLE)
        self.assertFalse(status["is_signing"])


if __name__ == "__main__":
    unittest.main()
