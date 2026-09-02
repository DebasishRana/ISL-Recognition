"""
Functionality: This converts MediaPipe hand landmarks into
a fixed (63+63) = 126 feature numerical representation. 

"""

import numpy as np


class HandFeatureExtractor:

    def __init__(self):
        self.features_per_hand = 21 * 3

        self.total_features = self.features_per_hand * 2

    def _normalize_hand(self, landmarks):

        points = np.array(
            [
                [landmark.x, landmark.y, landmark.z]
                for landmark in landmarks
            ],
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

        return points

    def _hand_to_features(self, landmarks):

        normalized_points = self._normalize_hand(
            landmarks
        )

        features = normalized_points.flatten()

        return features

    def extract(self, result):

        left_hand = np.zeros(
            self.features_per_hand,
            dtype=np.float32,
        )

        right_hand = np.zeros(
            self.features_per_hand,
            dtype=np.float32,
        )

        for index, landmarks in enumerate(
            result.hand_landmarks
        ):

            if index >= len(result.handedness):
                continue

            handedness_list = result.handedness[index]

            if not handedness_list:
                continue

            handedness = (
                handedness_list[0].category_name
            )

            features = self._hand_to_features(
                landmarks
            )


            if handedness == "Left":
                left_hand = features

            elif handedness == "Right":
                right_hand = features

        combined_features = np.concatenate(
            [
                left_hand,
                right_hand,
            ]
        )

        return combined_features