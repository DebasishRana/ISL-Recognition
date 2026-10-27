"""
Sign Activity Detector for Indian Sign Language Recognition.
Phase 1: Sign / Non-Sign Detection.

This module determines whether the user is actively signing or idle.
It distinguishes between:
  1. IDLE: No hands visible, hands resting on lap/desk, or hands held stationary.
  2. POSSIBLE_SIGN: Initial movement detected, accumulating temporal consistency.
  3. ACTIVE_SIGN: Confirmed, intentional sign gesture in progress.

Key Features:
- Differentiates resting/held hands from active signing.
- Computes hand velocity, centroid movement, and finger articulation.
- Uses shoulder-width normalization for scale and distance invariance.
- Fully configurable thresholds via SignActivityConfig.
- Ultra-lightweight computation (<0.1ms per frame on CPU).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np


class SignActivityState(str, Enum):
    """Activity states for sign detection."""
    IDLE = "IDLE"
    POSSIBLE_SIGN = "POSSIBLE_SIGN"
    ACTIVE_SIGN = "ACTIVE_SIGN"


@dataclass
class SignActivityConfig:
    """Configurable thresholds and parameters for sign activity detection."""

    # Velocity thresholds (normalized by shoulder span per second)
    motion_start_threshold: float = 0.90      # Velocity to enter POSSIBLE_SIGN from IDLE
    motion_active_threshold: float = 1.40     # Velocity to promote POSSIBLE_SIGN to ACTIVE_SIGN
    motion_continue_threshold: float = 0.60   # Lower hysteresis threshold to sustain ACTIVE_SIGN

    # Temporal consistency frame windows
    min_possible_frames: int = 2              # Consecutive moving frames required for POSSIBLE_SIGN
    min_active_frames: int = 3                # Consecutive active frames required for ACTIVE_SIGN
    cooldown_frames: int = 7                  # Hangover frames of still motion before dropping ACTIVE_SIGN
    possible_timeout_frames: int = 4          # Consecutive still frames before POSSIBLE_SIGN drops to IDLE

    # Position & Signing Space
    resting_y_threshold: float = 0.82         # Normalized Y screen coordinate below which hand is in resting zone
    resting_velocity_penalty: float = 0.50    # Discount factor applied to motion when hands are in resting zone

    # Articulation weighting
    articulation_weight: float = 0.45         # Weight given to finger articulation relative to wrist displacement
    smoothing_factor: float = 0.40            # EMA alpha smoothing factor (higher = more responsive, lower = smoother)

    # Scale fallback
    default_scale: float = 0.25               # Fallback scale if shoulders are not detected


@dataclass
class HandState:
    """Tracks position and kinematics of a single hand."""
    wrist: np.ndarray                         # [x, y, z]
    centroid: np.ndarray                      # [x, y, z]
    fingertips: np.ndarray                    # [5, 3] relative to wrist
    timestamp: float


class SignActivityDetector:
    """
    Dedicated sign activity detector that distinguishes IDLE, POSSIBLE_SIGN,
    and ACTIVE_SIGN states from MediaPipe tracking landmarks.
    """

    def __init__(self, config: Optional[SignActivityConfig] = None):
        self.config = config or SignActivityConfig()
        self.state = SignActivityState.IDLE

        # Historical kinematics
        self.prev_hands: Dict[str, HandState] = {}
        self.smoothed_velocity: float = 0.0

        # Temporal counters
        self.consecutive_moving_frames: int = 0
        self.consecutive_still_frames: int = 0
        self.active_frame_count: int = 0

        # Diagnostics
        self.last_raw_velocity: float = 0.0
        self.last_scale: float = self.config.default_scale
        self.hands_detected: int = 0
        self.in_signing_space: bool = False

    def reset(self):
        """Resets the detector state and history."""
        self.state = SignActivityState.IDLE
        self.prev_hands.clear()
        self.smoothed_velocity = 0.0
        self.consecutive_moving_frames = 0
        self.consecutive_still_frames = 0
        self.active_frame_count = 0
        self.last_raw_velocity = 0.0
        self.hands_detected = 0
        self.in_signing_space = False

    def _compute_scale(self, tracking_result: Dict[str, Any]) -> float:
        """
        Computes the normalization scale from pose shoulders.
        Ensures distance and camera invariance.
        """
        pose_result = tracking_result.get("pose")
        if not pose_result or not getattr(pose_result, "pose_landmarks", None):
            return self.config.default_scale

        landmarks = pose_result.pose_landmarks
        if not landmarks or len(landmarks[0]) < 13:
            return self.config.default_scale

        left_shoulder = landmarks[0][11]
        right_shoulder = landmarks[0][12]

        dx = left_shoulder.x - right_shoulder.x
        dy = left_shoulder.y - right_shoulder.y
        dz = left_shoulder.z - right_shoulder.z

        span = math.sqrt(dx * dx + dy * dy + dz * dz)
        if span > 0.05:
            return span

        return self.config.default_scale

    def _extract_hand_points(
        self,
        hand_landmarks
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts wrist, centroid, and relative fingertip positions for a hand.
        """
        points = np.empty((21, 3), dtype=np.float32)
        for i in range(21):
            lm = hand_landmarks[i]
            points[i, 0] = lm.x
            points[i, 1] = lm.y
            points[i, 2] = lm.z

        wrist = points[0].copy()
        centroid = np.mean(points, axis=0)

        # Fingertips: Thumb (4), Index (8), Middle (12), Ring (16), Pinky (20)
        fingertip_indices = (4, 8, 12, 16, 20)
        fingertips_rel = points[fingertip_indices, :] - wrist

        return wrist, centroid, fingertips_rel

    def _compute_hand_velocity(
        self,
        hand_label: str,
        wrist: np.ndarray,
        centroid: np.ndarray,
        fingertips: np.ndarray,
        timestamp: float,
        scale: float
    ) -> Tuple[float, bool]:
        """
        Computes instantaneous normalized velocity and signing-space status for a hand.
        """
        in_signing_space = wrist[1] < self.config.resting_y_threshold

        if hand_label not in self.prev_hands:
            # First frame for this hand: establish baseline, motion = 0
            self.prev_hands[hand_label] = HandState(
                wrist=wrist,
                centroid=centroid,
                fingertips=fingertips,
                timestamp=timestamp
            )
            return 0.0, in_signing_space

        prev = self.prev_hands[hand_label]
        dt = timestamp - prev.timestamp

        # Update stored state
        self.prev_hands[hand_label] = HandState(
            wrist=wrist,
            centroid=centroid,
            fingertips=fingertips,
            timestamp=timestamp
        )

        if dt <= 0.001 or dt > 0.6:
            # Invalid or stale time delta
            return 0.0, in_signing_space

        # Displacement of wrist and centroid
        wrist_dist = float(np.linalg.norm(wrist - prev.wrist))
        centroid_dist = float(np.linalg.norm(centroid - prev.centroid))
        macro_motion = max(wrist_dist, centroid_dist)

        # Articulation displacement (relative finger movement)
        articulation_dist = float(np.mean(np.linalg.norm(fingertips - prev.fingertips, axis=1)))

        total_motion = macro_motion + (self.config.articulation_weight * articulation_dist)

        # Penalize motion if the hand is in the resting/lap zone
        if not in_signing_space:
            total_motion *= self.config.resting_velocity_penalty

        # Normalized velocity: shoulder-spans per second
        norm_velocity = (total_motion / max(scale, 0.05)) / dt
        return norm_velocity, in_signing_space

    def process(
        self,
        tracking_result: Dict[str, Any],
        timestamp: float
    ) -> Dict[str, Any]:
        """
        Processes tracking results and updates the sign activity state.

        Returns a dictionary containing:
          - state: SignActivityState (IDLE, POSSIBLE_SIGN, ACTIVE_SIGN)
          - is_signing: bool (True only when state == ACTIVE_SIGN)
          - velocity: float (smoothed normalized velocity)
          - raw_velocity: float
          - hands_detected: int
          - in_signing_space: bool
          - active_frame_count: int
        """
        hands_result = tracking_result.get("hands")
        hand_landmarks_list = (
            getattr(hands_result, "hand_landmarks", [])
            if hands_result else []
        )

        self.hands_detected = len(hand_landmarks_list)
        scale = self._compute_scale(tracking_result)
        self.last_scale = scale

        # Case 1: No hands visible -> immediately IDLE
        if self.hands_detected == 0:
            self.prev_hands.clear()
            self.smoothed_velocity = 0.0
            self.consecutive_moving_frames = 0
            self.consecutive_still_frames += 1
            self.active_frame_count = 0
            self.in_signing_space = False
            self.state = SignActivityState.IDLE

            return self._build_status(raw_velocity=0.0)

        # Extract handedness labels if available
        handedness_list = (
            getattr(hands_result, "handedness", [])
            if hands_result else []
        )

        max_velocity = 0.0
        any_in_signing_space = False
        current_active_labels = set()

        for idx, landmarks in enumerate(hand_landmarks_list):
            if not landmarks or len(landmarks) < 21:
                continue

            # Determine hand label
            label = f"hand_{idx}"
            if idx < len(handedness_list) and handedness_list[idx]:
                cat = handedness_list[idx][0].category_name
                if cat in ("Left", "Right"):
                    label = cat

            current_active_labels.add(label)
            wrist, centroid, fingertips = self._extract_hand_points(landmarks)

            vel, in_space = self._compute_hand_velocity(
                hand_label=label,
                wrist=wrist,
                centroid=centroid,
                fingertips=fingertips,
                timestamp=timestamp,
                scale=scale
            )

            if vel > max_velocity:
                max_velocity = vel
            if in_space:
                any_in_signing_space = True

        # Prune hands that disappeared
        disappeared = set(self.prev_hands.keys()) - current_active_labels
        for label in disappeared:
            del self.prev_hands[label]

        self.in_signing_space = any_in_signing_space
        self.last_raw_velocity = max_velocity

        # Exponential Moving Average for velocity
        alpha = self.config.smoothing_factor
        self.smoothed_velocity = (
            alpha * max_velocity + (1.0 - alpha) * self.smoothed_velocity
        )

        # State transition logic
        self._update_state_machine()

        return self._build_status(raw_velocity=max_velocity)

    def _update_state_machine(self):
        """Updates IDLE, POSSIBLE_SIGN, and ACTIVE_SIGN state transitions."""
        v = self.smoothed_velocity

        if v >= self.config.motion_start_threshold:
            # Motion detected
            self.consecutive_moving_frames += 1
            self.consecutive_still_frames = 0
        else:
            # Still / below threshold
            self.consecutive_still_frames += 1
            self.consecutive_moving_frames = 0

        # State 1: IDLE
        if self.state == SignActivityState.IDLE:
            self.active_frame_count = 0
            if (
                self.consecutive_moving_frames >= self.config.min_possible_frames
                and self.in_signing_space
            ):
                self.state = SignActivityState.POSSIBLE_SIGN

        # State 2: POSSIBLE_SIGN
        elif self.state == SignActivityState.POSSIBLE_SIGN:
            if self.consecutive_still_frames >= self.config.possible_timeout_frames:
                # Movement ceased before becoming an active sign
                self.state = SignActivityState.IDLE
                self.consecutive_moving_frames = 0
            elif (
                self.consecutive_moving_frames >= self.config.min_active_frames
                and v >= self.config.motion_active_threshold
                and self.in_signing_space
            ):
                # Sustained intentional movement confirmed
                self.state = SignActivityState.ACTIVE_SIGN
                self.active_frame_count = self.consecutive_moving_frames

        # State 3: ACTIVE_SIGN
        elif self.state == SignActivityState.ACTIVE_SIGN:
            if v >= self.config.motion_continue_threshold:
                # Active motion continues
                self.active_frame_count += 1
                self.consecutive_still_frames = 0
            else:
                # Motion dropped below continuation threshold (potential sign hold or rest)
                if self.consecutive_still_frames >= self.config.cooldown_frames:
                    # Stillness sustained beyond hangover window: return to IDLE
                    self.state = SignActivityState.IDLE
                    self.active_frame_count = 0
                    self.consecutive_moving_frames = 0
                else:
                    # In hold/apex cooldown window: sustain ACTIVE_SIGN
                    self.active_frame_count += 1

    def _build_status(self, raw_velocity: float) -> Dict[str, Any]:
        """Constructs the output status dictionary."""
        return {
            "state": self.state,
            "is_signing": self.state == SignActivityState.ACTIVE_SIGN,
            "velocity": round(self.smoothed_velocity, 3),
            "raw_velocity": round(raw_velocity, 3),
            "hands_detected": self.hands_detected,
            "in_signing_space": self.in_signing_space,
            "active_frame_count": self.active_frame_count,
        }

    def get_state(self) -> SignActivityState:
        """Returns the current state."""
        return self.state

    def is_signing(self) -> bool:
        """Returns True if meaningful signing activity is confirmed."""
        return self.state == SignActivityState.ACTIVE_SIGN
