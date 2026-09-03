"""
Functionality: This script determines which video archives are required for our 50-sign ISL dataset.

"""

from pathlib import Path
import re

import pandas as pd


METADATA_DIR = Path(
    r"D:\ISL-Dataset\metadata"
)

DATASET_METADATA_FILE = (
    METADATA_DIR / "dataset_metadata.csv"
)

SOURCE_FILES = (
    METADATA_DIR / "video_source_files.csv"
)

OUTPUT_FILE = (
    METADATA_DIR / "required_video_files.csv"
)


def format_size(size_bytes):
    size = float(size_bytes)

    units = [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]

    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size:.2f} PB"


def get_archive_category(filename):
    match = re.match(
        r"^(.*?)_\d+of\d+f?\.zip$",
        filename,
        re.IGNORECASE,
    )

    if match:
        return match.group(1)

    return None


def main():

    print("=" * 60)
    print("REQUIRED VIDEO FILE SELECTION")
    print("=" * 60)

    if not DATASET_METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset metadata not found:\n"
            f"{DATASET_METADATA_FILE}"
        )

    if not SOURCE_FILES.exists():
        raise FileNotFoundError(
            f"Video source information not found:\n"
            f"{SOURCE_FILES}"
        )

    print()
    print("Reading dataset metadata...")

    dataset_metadata = pd.read_csv(
        DATASET_METADATA_FILE
    )

    if "parent_label" not in dataset_metadata.columns:
        raise RuntimeError(
            "The dataset metadata does not contain "
            "'parent_label'."
        )

    required_categories = sorted(
        dataset_metadata["parent_label"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    print()
    print(
        "Required categories:",
        len(required_categories)
    )

    for category in required_categories:
        print(
            f"  - {category}"
        )

    print()
    print("Reading available video archives...")

    source_files = pd.read_csv(
        SOURCE_FILES
    )

    if "file_name" not in source_files.columns:
        raise RuntimeError(
            "The source file list does not contain "
            "'file_name'."
        )

    archives = source_files[
        source_files["file_name"]
        .astype(str)
        .str.lower()
        .str.endswith(".zip")
    ].copy()

    archives["file_name"] = (
        archives["file_name"]
        .astype(str)
        .str.strip()
    )

    archives["category"] = archives[
        "file_name"
    ].apply(
        get_archive_category
    )

    required_archives = archives[
        archives["category"].isin(
            required_categories
        )
    ].copy()

    found_categories = set(
        required_archives["category"]
        .dropna()
    )

    missing_categories = [
        category
        for category in required_categories
        if category not in found_categories
    ]

    if missing_categories:
        print()
        print("=" * 60)
        print("WARNING: MISSING CATEGORIES")
        print("=" * 60)

        for category in missing_categories:
            print(
                f"  - {category}"
            )

    total_size = required_archives[
        "size_bytes"
    ].sum()

    required_archives.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 60)
    print("REQUIRED VIDEO ARCHIVES")
    print("=" * 60)

    print()

    for _, row in required_archives.iterrows():
        print(
            f"- {row['file_name']}"
        )

        print(
            f"  Size: {row['size_readable']}"
        )

    print()
    print("=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    print()
    print(
        "Required archives:",
        len(required_archives)
    )

    print(
        "Total download size:",
        format_size(total_size)
    )

    print()
    print("Selection saved to:")
    print(OUTPUT_FILE)

    print()
    print("=" * 60)
    print("SELECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()