"""
Functionality:
This module provides the single unified feature extractor
for the ISL recognition system.

It converts MediaPipe hand, pose, and face landmarks into
one fixed-size feature vector.

Features:
    Hands: 2 * 21 * 3 = 126
    Pose: 33 * 3 = 99
    Face: 478 * 3 = 1434
    Presence indicators: 3

Total:
    126 + 99 + 1434 + 3 = 1662 features per frame

Missing modalities are represented by zero-filled features
and an explicit presence indicator.

The extractor is designed so that a sign does not fail when
face or another supported modality is unavailable.
"""

import numpy as np


class FeatureExtractor:
    """
    Unified feature extractor for hands, pose, and face.
    """

    HAND_FEATURES = 126
    POSE_FEATURES = 99
    FACE_FEATURES = 1434
    PRESENCE_FEATURES = 3

    TOTAL_FEATURES = (
        HAND_FEATURES
        + POSE_FEATURES
        + FACE_FEATURES
        + PRESENCE_FEATURES
    )

    def __init__(self):
        self.hand_features = self.HAND_FEATURES
        self.pose_features = self.POSE_FEATURES
        self.face_features = self.FACE_FEATURES
        self.total_features = self.TOTAL_FEATURES

    def _normalize_hand(self, landmarks):
        points = np.array(
            [
                [landmark.x, landmark.y, landmark.z]
                for landmark in landmarks
            ],
            dtype=np.float32,
        )

        if len(points) != 21:
            return np.zeros(
                self.HAND_FEATURES // 2,
                dtype=np.float32,
            )

        wrist = points[0].copy()
        points = points - wrist

        distances = np.linalg.norm(
            points,
            axis=1,
        )

        scale = np.max(distances)

        if scale > 1e-6:
            points = points / scale

        return points.flatten()

    def _normalize_pose(self, landmarks):
        points = np.array(
            [
                [landmark.x, landmark.y, landmark.z]
                for landmark in landmarks
            ],
            dtype=np.float32,
        )

        if len(points) != 33:
            return np.zeros(
                self.POSE_FEATURES,
                dtype=np.float32,
            )

        hip_center = (
            points[23] + points[24]
        ) / 2.0

        points = points - hip_center

        shoulder_distance = np.linalg.norm(
            points[11] - points[12]
        )

        if shoulder_distance > 1e-6:
            points = (
                points
                / shoulder_distance
            )

        return points.flatten()

    def _normalize_face(self, landmarks):
        points = np.array(
            [
                [landmark.x, landmark.y, landmark.z]
                for landmark in landmarks
            ],
            dtype=np.float32,
        )

        if len(points) != 478:
            return np.zeros(
                self.FACE_FEATURES,
                dtype=np.float32,
            )

        nose = points[1].copy()
        points = points - nose

        left_eye = points[33]
        right_eye = points[263]

        eye_distance = np.linalg.norm(
            left_eye - right_eye
        )

        if eye_distance > 1e-6:
            points = (
                points
                / eye_distance
            )

        return points.flatten()

    def _extract_hands(self, result):
        left_hand = np.zeros(
            63,
            dtype=np.float32,
        )

        right_hand = np.zeros(
            63,
            dtype=np.float32,
        )

        hands_present = 0.0

        hands_result = result["hands"]

        for index, landmarks in enumerate(
            hands_result.hand_landmarks
        ):
            if index >= len(
                hands_result.handedness
            ):
                continue

            handedness_list = (
                hands_result.handedness[index]
            )

            if not handedness_list:
                continue

            handedness = (
                handedness_list[0].category_name
            )

            features = self._normalize_hand(
                landmarks
            )

            if handedness == "Left":
                left_hand = features
                hands_present = 1.0

            elif handedness == "Right":
                right_hand = features
                hands_present = 1.0

        return (
            left_hand,
            right_hand,
            hands_present,
        )

    def _extract_pose(self, result):
        pose_result = result["pose"]

        if not pose_result.pose_landmarks:
            return (
                np.zeros(
                    self.POSE_FEATURES,
                    dtype=np.float32,
                ),
                0.0,
            )

        pose_features = (
            self._normalize_pose(
                pose_result.pose_landmarks[0]
            )
        )

        return pose_features, 1.0

    def _extract_face(self, result):
        face_result = result["face"]

        if not face_result.face_landmarks:
            return (
                np.zeros(
                    self.FACE_FEATURES,
                    dtype=np.float32,
                ),
                0.0,
            )

        face_features = (
            self._normalize_face(
                face_result.face_landmarks[0]
            )
        )

        return face_features, 1.0

    def extract(self, result):
        (
            left_hand,
            right_hand,
            hands_present,
        ) = self._extract_hands(result)

        pose_features, pose_present = (
            self._extract_pose(result)
        )

        face_features, face_present = (
            self._extract_face(result)
        )

        presence = np.array(
            [
                hands_present,
                pose_present,
                face_present,
            ],
            dtype=np.float32,
        )

        combined = np.concatenate(
            [
                left_hand,
                right_hand,
                pose_features,
                face_features,
                presence,
            ]
        )

        if combined.shape != (
            self.total_features,
        ):
            raise ValueError(
                f"Unexpected feature shape: "
                f"{combined.shape}; "
                f"expected "
                f"({self.total_features},)"
            )

        if not np.isfinite(
            combined
        ).all():
            raise ValueError(
                "Feature vector contains "
                "NaN or Inf values."
            )

        return combined