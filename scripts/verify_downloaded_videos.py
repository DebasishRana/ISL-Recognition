"""
Functionality: This script verifies that all 958 selected dataset videos exist inside the 44 downloaded ZIP archives.

"""

from pathlib import Path
from zipfile import ZipFile

import pandas as pd


DATASET_ROOT = Path("D:/ISL-Dataset")
METADATA_ROOT = DATASET_ROOT / "metadata"
VIDEO_ROOT = DATASET_ROOT / "videos"

DATASET_METADATA_FILE = METADATA_ROOT / "dataset_metadata.csv"
SOURCE_FILE = METADATA_ROOT / "video_source_files.csv"
MAPPING_FILE = METADATA_ROOT / "archive_video_mapping.csv"


def normalize_path(path):
    return str(path).replace("\\", "/").lstrip("./")


def load_required_videos():
    dataframe = pd.read_csv(
        DATASET_METADATA_FILE
    )

    if "video_path" not in dataframe.columns:
        raise KeyError(
            "video_path column not found in dataset metadata."
        )

    return dataframe


def load_archives():
    dataframe = pd.read_csv(
        SOURCE_FILE
    )

    if "file_name" not in dataframe.columns:
        raise KeyError(
            "file_name column not found in source metadata."
        )

    archives = dataframe[
        dataframe["file_name"]
        .astype(str)
        .str.lower()
        .str.endswith(".zip")
    ].copy()

    return archives["file_name"].astype(str).tolist()


def scan_archives(required_videos, archives):
    required_paths = {
        normalize_path(path)
        for path in required_videos["video_path"]
    }

    mapping = {}

    print()
    print("SCANNING DOWNLOADED ARCHIVES")
    print("=" * 60)
    print(f"Archives to scan : {len(archives)}")
    print(f"Required videos  : {len(required_paths)}")
    print()

    for index, archive_name in enumerate(
        archives,
        start=1,
    ):
        archive_path = VIDEO_ROOT / archive_name

        if not archive_path.exists():
            print(
                f"[{index}/{len(archives)}] MISSING: "
                f"{archive_name}"
            )
            continue

        print(
            f"[{index}/{len(archives)}] Scanning: "
            f"{archive_name}"
        )

        with ZipFile(
            archive_path,
            "r",
        ) as archive:

            names = archive.namelist()

        normalized_names = {
            normalize_path(name): name
            for name in names
        }

        for required_path in required_paths:
            if required_path in normalized_names:
                mapping[required_path] = {
                    "archive": archive_name,
                    "archive_path": normalized_names[
                        required_path
                    ],
                }

    return mapping


def main():
    required_videos = load_required_videos()
    archives = load_archives()

    mapping = scan_archives(
        required_videos,
        archives,
    )

    required_paths = {
        normalize_path(path)
        for path in required_videos["video_path"]
    }

    found_paths = set(mapping.keys())
    missing_paths = required_paths - found_paths

    print()
    print("=" * 60)
    print("VIDEO VERIFICATION RESULT")
    print("=" * 60)

    print(
        f"Required videos : {len(required_paths)}"
    )

    print(
        f"Videos found    : {len(found_paths)}"
    )

    print(
        f"Videos missing  : {len(missing_paths)}"
    )

    if missing_paths:
        print()
        print("MISSING VIDEOS")
        print("-" * 60)

        for path in sorted(missing_paths):
            print(path)

        print()
        print(
            "Verification FAILED."
        )

        return

    mapping_rows = []

    for _, row in required_videos.iterrows():
        video_path = normalize_path(
            row["video_path"]
        )

        archive_info = mapping[video_path]

        mapping_rows.append(
            {
                "video_path": video_path,
                "archive": archive_info["archive"],
                "archive_path": archive_info["archive_path"],
                "label": row["label"],
                "parent_label": row["parent_label"],
                "split": row["split"],
            }
        )

    mapping_dataframe = pd.DataFrame(
        mapping_rows
    )

    mapping_dataframe.to_csv(
        MAPPING_FILE,
        index=False,
    )

    print()
    print("Verification PASSED.")
    print()
    print(
        f"Mapping saved to:"
    )
    print(MAPPING_FILE)

    print()
    print("ARCHIVE DISTRIBUTION")
    print("-" * 60)

    distribution = (
        mapping_dataframe["archive"]
        .value_counts()
        .sort_index()
    )

    for archive, count in distribution.items():
        print(
            f"{archive}: {count} videos"
        )


if __name__ == "__main__":
    main()