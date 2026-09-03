"""
Functionality: verifies MediaPipe hand detection results and convertion into 126 feature

"""

import cv2
import numpy as np

from hand_tracker import HandTracker
from feature_extractor import HandFeatureExtractor


def main():

    print("Starting feature extraction test...")
    print("Press Q to quit.")

    tracker = HandTracker()

    extractor = HandFeatureExtractor()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        tracker.close()
        raise RuntimeError(
            "Could not open webcam."
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720,
    )

    frame_count = 0

    diagnostic_printed = False

    try:

        while True:


            success, frame = cap.read()

            if not success:
                print("Failed to read webcam frame.")
                break

            frame = cv2.flip(frame, 1)


            timestamp_ms = frame_count * 33
            frame_count += 1


            result = tracker.process(
                frame,
                timestamp_ms,
            )


            features = extractor.extract(result)

            num_hands = len(
                result.hand_landmarks
            )


            if num_hands > 0 and not diagnostic_printed:

                print()
                print("=" * 55)
                print("FEATURE EXTRACTION TEST")
                print("=" * 55)

                print(
                    "Number of detected hands:",
                    num_hands,
                )

                print(
                    "Feature shape:",
                    features.shape,
                )

                print(
                    "Feature count:",
                    len(features),
                )

                print(
                    "Data type:",
                    features.dtype,
                )

                print(
                    "Contains NaN:",
                    np.isnan(features).any(),
                )

                print(
                    "Contains Inf:",
                    np.isinf(features).any(),
                )

                print(
                    "Non-zero features:",
                    np.count_nonzero(features),
                )

                print()
                print("First 15 features:")

                print(features[:15])

                print("=" * 55)
                print()

                diagnostic_printed = True


            cv2.putText(
                frame,
                f"Hands: {num_hands}",
                (20, frame.shape[0] - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Features: {len(features)}",
                (20, frame.shape[0] - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                "Q: Quit",
                (20, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "ISL Feature Extraction Test",
                frame,
            )


            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        cap.release()

        tracker.close()

        cv2.destroyAllWindows()

        print("Feature extraction test finished.")


if __name__ == "__main__":
    main()