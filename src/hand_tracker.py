"""
Functionality: This module provides the HandTracker class.

"""


from pathlib import Path

import cv2
import mediapipe as mp


class HandTracker:


    def __init__(self):
        project_root = Path(__file__).resolve().parents[1]

        self.model_path = project_root / "models" / "hand_landmarker.task"

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found at: {self.model_path}"
            )


        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(self.model_path)
            ),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = HandLandmarker.create_from_options(options)

    def process(self, frame, timestamp_ms):


        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )


        result = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        return result

    def close(self):

        self.landmarker.close()