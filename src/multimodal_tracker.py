"""
Functionality: The tracker processes video frames using three MediaPipe landmark models:
1. Hand Landmarker
2. Pose Landmarker
3. Face Landmarker

"""

from pathlib import Path

import cv2
import mediapipe as mp


class MultimodalTracker:
    """
    MediaPipe tracker for hands, pose, and face landmarks.
    """

    def __init__(self):
        project_root = Path(__file__).resolve().parents[1]
        models_dir = project_root / "models"

        self.hand_model_path = models_dir / "hand_landmarker.task"
        self.pose_model_path = models_dir / "pose_landmarker_lite.task"
        self.face_model_path = models_dir / "face_landmarker.task"

        required_models = [
            self.hand_model_path,
            self.pose_model_path,
            self.face_model_path,
        ]

        for model_path in required_models:
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found: {model_path}"
                )

        BaseOptions = mp.tasks.BaseOptions
        vision = mp.tasks.vision

        self.hand_landmarker = (
            vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=str(
                            self.hand_model_path
                        )
                    ),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=2,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            )
        )

        self.pose_landmarker = (
            vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=str(
                            self.pose_model_path
                        )
                    ),
                    running_mode=vision.RunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            )
        )

        self.face_landmarker = (
            vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=str(
                            self.face_model_path
                        )
                    ),
                    running_mode=vision.RunningMode.VIDEO,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                )
            )
        )

    def process(self, frame, timestamp_ms):
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        hand_result = self.hand_landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        pose_result = self.pose_landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        face_result = self.face_landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        return {
            "hands": hand_result,
            "pose": pose_result,
            "face": face_result,
        }

    def close(self):
        self.hand_landmarker.close()
        self.pose_landmarker.close()
        self.face_landmarker.close()