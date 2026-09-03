"""
Functionality: This script checks the selected dataset metadata for duplicate video paths.

"""

from pathlib import Path

import pandas as pd


DATASET_ROOT = Path("D:/ISL-Dataset")
METADATA_FILE = DATASET_ROOT / "metadata" / "dataset_metadata.csv"


def main():
    dataframe = pd.read_csv(METADATA_FILE)

    total_rows = len(dataframe)
    unique_paths = dataframe["video_path"].nunique()
    duplicate_rows = total_rows - unique_paths

    print()
    print("METADATA DUPLICATE CHECK")
    print("=" * 60)

    print(f"Total metadata rows : {total_rows}")
    print(f"Unique video paths  : {unique_paths}")
    print(f"Duplicate rows      : {duplicate_rows}")

    print()

    if duplicate_rows == 0:
        print("No duplicate video paths found.")

    else:
        duplicates = dataframe[
            dataframe["video_path"].duplicated(
                keep=False
            )
        ].sort_values("video_path")

        print("DUPLICATE VIDEO PATHS")
        print("-" * 60)

        for video_path, group in duplicates.groupby(
            "video_path"
        ):
            print()
            print(video_path)

            for _, row in group.iterrows():
                print(
                    f"  label={row['label']} | "
                    f"parent={row['parent_label']} | "
                    f"split={row['split']}"
                )

        print()
        print(
            f"Number of duplicated metadata rows: "
            f"{duplicate_rows}"
        )


if __name__ == "__main__":
    main()