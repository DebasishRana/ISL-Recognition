"""
Functionality: It automatically detects filename and size columns, compares source and selected archives, and reports excluded archives and exact storage requirements.
"""

from pathlib import Path

import pandas as pd


DATASET_ROOT = Path("D:/ISL-Dataset")
METADATA_ROOT = DATASET_ROOT / "metadata"

SOURCE_FILE = METADATA_ROOT / "video_source_files.csv"
SELECTED_FILE = METADATA_ROOT / "required_video_files.csv"
DATASET_METADATA_FILE = METADATA_ROOT / "dataset_metadata.csv"


def find_column(dataframe, candidates):
    for column in candidates:
        if column in dataframe.columns:
            return column

    raise KeyError(
        f"Could not find any of these columns: {candidates}\n"
        f"Available columns: {list(dataframe.columns)}"
    )


def main():
    source = pd.read_csv(SOURCE_FILE)
    selected = pd.read_csv(SELECTED_FILE)
    dataset = pd.read_csv(DATASET_METADATA_FILE)

    source_name_column = find_column(
        source,
        ["name", "filename", "file_name", "key"],
    )

    selected_name_column = find_column(
        selected,
        ["name", "filename", "file_name", "key"],
    )

    source_size_column = find_column(
        source,
        ["size", "size_bytes", "file_size"],
    )

    selected_size_column = find_column(
        selected,
        ["size", "size_bytes", "file_size"],
    )

    source_archives = source[
        source[source_name_column]
        .astype(str)
        .str.lower()
        .str.endswith(".zip")
    ].copy()

    selected_archives = selected[
        selected[selected_name_column]
        .astype(str)
        .str.lower()
        .str.endswith(".zip")
    ].copy()

    source_names = set(
        source_archives[source_name_column].astype(str)
    )

    selected_names = set(
        selected_archives[selected_name_column].astype(str)
    )

    excluded_names = source_names - selected_names

    source_size = source_archives[source_size_column].sum()
    selected_size = selected_archives[selected_size_column].sum()
    difference = source_size - selected_size

    required_categories = sorted(
        dataset["video_path"]
        .astype(str)
        .str.split("/")
        .str[0]
        .unique()
    )

    print()
    print("VIDEO ARCHIVE VERIFICATION")
    print("=" * 60)

    print(f"Source CSV filename column   : {source_name_column}")
    print(f"Selected CSV filename column : {selected_name_column}")
    print()

    print(f"Source ZIP archives   : {len(source_archives)}")
    print(f"Selected ZIP archives : {len(selected_archives)}")

    print()
    print(f"Source size   : {source_size:,} bytes")
    print(f"Selected size : {selected_size:,} bytes")
    print(f"Difference    : {difference:,} bytes")

    print()
    print("EXCLUDED ZIP ARCHIVES")
    print("-" * 60)

    if excluded_names:
        for name in sorted(excluded_names):
            row = source_archives[
                source_archives[source_name_column].astype(str) == name
            ]

            size = int(row[source_size_column].iloc[0])

            print(f"{name}")
            print(f"Size: {size:,} bytes")
            print()
    else:
        print("None")

    print("REQUIRED DATASET CATEGORIES")
    print("-" * 60)

    for category in required_categories:
        print(category)

    print()
    print("VERIFICATION")
    print("-" * 60)

    if excluded_names:
        print(
            f"{len(excluded_names)} ZIP archive(s) are excluded "
            "from the current selection."
        )
    else:
        print("All source ZIP archives are selected.")

    print()
    print("No files have been downloaded by this script.")


if __name__ == "__main__":
    main()