"""
Functionality:
-------------
This script tests multimodal landmark detection on 15 videos.

It:
1. Reads the cleaned metadata.
2. Resolves each video to its ZIP archive.
3. Temporarily extracts the required video.
4. Processes up to 30 frames using the multimodal tracker.
5. Measures hand, pose, and face detection rates.
6. Checks for processing failures.
7. Saves a compact CSV report.
8. Prints only a short summary to the terminal.
"""

from pathlib import Path
import sys
import tempfile
import time

import cv2
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src"),
)

from extract_landmarks import LandmarkExtractor
from multimodal_tracker import MultimodalTracker


DATASET_ROOT = Path(r"D:\ISL-Dataset")

METADATA_PATH = (
    DATASET_ROOT
    / "metadata"
    / "dataset_metadata_clean.csv"
)

OUTPUT_PATH = (
    DATASET_ROOT
    / "extracted"
    / "multimodal_test_report.csv"
)

SAMPLE_SIZE = 15
MAX_FRAMES_PER_VIDEO = 30


def normalize_path(value):
    return str(value).replace("\\", "/").strip().lower()


def find_archive(mapping, video_path):
    target = normalize_path(video_path)

    for mapped_video, archive_file in mapping.items():
        if normalize_path(mapped_video) == target:
            return archive_file

    return None


def process_video(tracker, video_path, max_frames):
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        return {
            "status": "failed_to_open",
            "frames": 0,
            "hand_frames": 0,
            "pose_frames": 0,
            "face_frames": 0,
            "processing_seconds": 0.0,
        }

    frame_count = 0
    hand_frames = 0
    pose_frames = 0
    face_frames = 0

    start_time = time.perf_counter()

    while frame_count < max_frames:
        success, frame = capture.read()

        if not success:
            break

        timestamp_ms = int(frame_count * 33.333)

        result = tracker.process(
            frame,
            timestamp_ms,
        )

        hands = result["hands"]
        pose = result["pose"]
        face = result["face"]

        if len(hands.hand_landmarks) > 0:
            hand_frames += 1

        if len(pose.pose_landmarks) > 0:
            pose_frames += 1

        if len(face.face_landmarks) > 0:
            face_frames += 1

        frame_count += 1

    processing_seconds = (
        time.perf_counter() - start_time
    )

    capture.release()

    status = "success"

    if frame_count == 0:
        status = "failed_to_read"

    return {
        "status": status,
        "frames": frame_count,
        "hand_frames": hand_frames,
        "pose_frames": pose_frames,
        "face_frames": face_frames,
        "processing_seconds": processing_seconds,
    }


def main():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    sample = metadata.sample(
        n=min(SAMPLE_SIZE, len(metadata)),
        random_state=42,
    )

    source = LandmarkExtractor()
    mapping = source.load_mapping()

    results = []

    print()
    print("Multimodal extraction test")
    print("==========================")
    print(
        f"Testing {len(sample)} videos..."
    )
    print()

    for position, (_, row) in enumerate(
        sample.iterrows(),
        start=1,
    ):
        video_path = str(
            row["video_path"]
        )

        archive_file = find_archive(
            mapping,
            video_path,
        )

        label = str(row["label"])

        display_name = Path(
            video_path
        ).name

        print(
            f"[{position}/{len(sample)}] "
            f"{label} / {display_name}",
            end=" ... ",
            flush=True,
        )

        if archive_file is None:
            print("NO ARCHIVE")

            results.append(
                {
                    "video_path": video_path,
                    "label": label,
                    "status": "archive_not_found",
                    "frames": 0,
                    "hand_detection_rate": 0.0,
                    "pose_detection_rate": 0.0,
                    "face_detection_rate": 0.0,
                    "processing_seconds": 0.0,
                }
            )

            continue

        archive_path = source.locate_archive(
            archive_file
        )

        if archive_path is None:
            print("ARCHIVE MISSING")

            results.append(
                {
                    "video_path": video_path,
                    "label": label,
                    "status": "archive_missing",
                    "frames": 0,
                    "hand_detection_rate": 0.0,
                    "pose_detection_rate": 0.0,
                    "face_detection_rate": 0.0,
                    "processing_seconds": 0.0,
                }
            )

            continue

        tracker = None

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                extracted_video = (
                    source.extract_video_from_zip(
                        archive_path,
                        video_path,
                        Path(temp_dir),
                    )
                )

                tracker = MultimodalTracker()

                result = process_video(
                    tracker,
                    extracted_video,
                    MAX_FRAMES_PER_VIDEO,
                )

            frames = result["frames"]

            if frames > 0:
                hand_rate = (
                    result["hand_frames"]
                    / frames
                )

                pose_rate = (
                    result["pose_frames"]
                    / frames
                )

                face_rate = (
                    result["face_frames"]
                    / frames
                )
            else:
                hand_rate = 0.0
                pose_rate = 0.0
                face_rate = 0.0

            results.append(
                {
                    "video_path": video_path,
                    "label": label,
                    "status": result["status"],
                    "frames": frames,
                    "hand_detection_rate": hand_rate,
                    "pose_detection_rate": pose_rate,
                    "face_detection_rate": face_rate,
                    "processing_seconds": result[
                        "processing_seconds"
                    ],
                }
            )

            if result["status"] == "success":
                print(
                    f"OK | "
                    f"H {hand_rate:.0%} | "
                    f"P {pose_rate:.0%} | "
                    f"F {face_rate:.0%}"
                )
            else:
                print(result["status"])

        except Exception as error:
            print("ERROR")

            results.append(
                {
                    "video_path": video_path,
                    "label": label,
                    "status": f"error: {type(error).__name__}",
                    "frames": 0,
                    "hand_detection_rate": 0.0,
                    "pose_detection_rate": 0.0,
                    "face_detection_rate": 0.0,
                    "processing_seconds": 0.0,
                }
            )

        finally:
            if tracker is not None:
                tracker.close()

    report = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    successful = (
        report["status"] == "success"
    ).sum()

    print()
    print("SUMMARY")
    print("=======")
    print(
        f"Videos tested : {len(report)}"
    )
    print(
        f"Successful    : {successful}"
    )
    print(
        f"Failed        : {len(report) - successful}"
    )

    if successful > 0:
        successful_report = report[
            report["status"] == "success"
        ]

        print(
            f"Avg hands     : "
            f"{successful_report['hand_detection_rate'].mean():.1%}"
        )

        print(
            f"Avg pose      : "
            f"{successful_report['pose_detection_rate'].mean():.1%}"
        )

        print(
            f"Avg face      : "
            f"{successful_report['face_detection_rate'].mean():.1%}"
        )

        print(
            f"Avg time      : "
            f"{successful_report['processing_seconds'].mean():.2f}s/video"
        )

    print()
    print(
        f"Report saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()