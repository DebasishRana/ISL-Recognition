"""
Functionality: This script tests the complete webcam and MediaPipe hand-tracking pipeline.

"""

import time
import cv2
from hand_tracker import HandTracker



HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def draw_hand_landmarks(frame, hand_landmarks):

    height, width = frame.shape[:2]

    points = []

    for landmark in hand_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2,
        )

    for x, y in points:
        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 0, 255),
            -1,
        )


def main():

    print("Starting ISL hand-tracking test...")
    print("Press Q to quit.")

    tracker = HandTracker()

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "ISL Hand Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    fullscreen = False

    if not cap.isOpened():
        tracker.close()
        raise RuntimeError(
            "Could not open webcam. "
            "Check that your camera is connected and available."
        )


    timestamp_frame_count = 0

    fps_frame_count = 0

    fps_start_time = time.perf_counter()
    fps = 0.0

    try:
        while True:

            success, frame = cap.read()

            if not success:
                print("Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)

            timestamp_ms = timestamp_frame_count * 33
            timestamp_frame_count += 1

            result = tracker.process(
                frame,
                timestamp_ms,
            )

            num_hands = len(result.hand_landmarks)

            for hand_landmarks in result.hand_landmarks:
                draw_hand_landmarks(
                    frame,
                    hand_landmarks,
                )


            fps_frame_count += 1

            elapsed = time.perf_counter() - fps_start_time

            if elapsed >= 1.0:

                fps = fps_frame_count / elapsed

                fps_frame_count = 0
                fps_start_time = time.perf_counter()

            cv2.putText(
                frame,
                f"Hands detected: {num_hands}",
                (20, frame.shape[0] - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, frame.shape[0] - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                "Press Q to quit",
                (20, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                window_name,
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("f"):
                fullscreen = not fullscreen

                if fullscreen:
                    cv2.setWindowProperty(
                        window_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN,
                    )
                else:
                    cv2.setWindowProperty(
                        window_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_NORMAL,
                    )

                    cv2.resizeWindow(
                        window_name,
                        1280,
                        720,
                    )


    finally:
        cap.release()

        tracker.close()

        cv2.destroyAllWindows()

        print("Hand-tracking test finished.")


if __name__ == "__main__":
    main()