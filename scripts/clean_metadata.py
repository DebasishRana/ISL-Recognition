"""
Functionality: This module validates and cleans the ISL dataset metadata.

"""

from pathlib import Path

import pandas as pd


class MetadataCleaner:
    def __init__(self):
        project_root = Path(__file__).resolve().parents[1]

        self.dataset_root = Path(r"D:\ISL-Dataset")

        self.metadata_path = (
            self.dataset_root
            / "metadata"
            / "dataset_metadata.csv"
        )

        self.output_path = (
            self.dataset_root
            / "metadata"
            / "dataset_metadata_clean.csv"
        )

        self.report_path = (
            self.dataset_root
            / "metadata"
            / "metadata_quality_report.csv"
        )

    def load_metadata(self):
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}"
            )

        dataframe = pd.read_csv(self.metadata_path)

        required_columns = {
            "parent_label",
            "label",
            "video_path",
            "split",
        }

        missing_columns = required_columns - set(
            dataframe.columns
        )

        if missing_columns:
            raise ValueError(
                "Missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        return dataframe

    def parse_video_path(self, video_path):
        path = Path(str(video_path))

        parts = path.parts

        if len(parts) < 3:
            return None, None, None

        parent_label = parts[-3]
        label = parts[-2]
        filename = parts[-1]

        return parent_label, label, filename

    def validate_row(self, row):
        expected_parent, expected_label, filename = (
            self.parse_video_path(row["video_path"])
        )

        if expected_parent is None:
            return False, "invalid_path_structure"

        parent_match = (
            str(row["parent_label"]).strip()
            == str(expected_parent).strip()
        )

        label_match = (
            str(row["label"]).strip()
            == str(expected_label).strip()
        )

        if parent_match and label_match:
            return True, "path_metadata_match"

        if parent_match and not label_match:
            return False, "label_mismatch"

        if not parent_match and label_match:
            return False, "parent_mismatch"

        return False, "parent_and_label_mismatch"
#change
    def clean_duplicates(self, dataframe):
        duplicate_mask = dataframe.duplicated(
            subset=["video_path"],
            keep=False,
        )

        duplicate_dataframe = dataframe[
            duplicate_mask
        ].copy()

        duplicate_paths = (
            duplicate_dataframe["video_path"]
            .drop_duplicates()
            .tolist()
        )

        kept_indices = set()
        removed_indices = set()
        ambiguous_indices = set()
        report_rows = []

        for video_path in duplicate_paths:
            group = duplicate_dataframe[
                duplicate_dataframe["video_path"]
                == video_path
            ]

            metadata_columns = [
                "parent_label",
                "label",
                "video_path",
                "include_50",
                "split",
            ]

            identical_metadata = (
                group[metadata_columns]
                .drop_duplicates()
            )

            if len(identical_metadata) == 1:
                kept_index = group.index[0]

                kept_indices.add(kept_index)

                for index in group.index:
                    if index != kept_index:
                        removed_indices.add(index)

                report_rows.append(
                    {
                        "video_path": video_path,
                        "decision": "deduplicated",
                        "kept_rows": 1,
                        "removed_rows": len(group) - 1,
                        "reason": "identical_metadata_rows",
                    }
                )

                continue

            matches = []

            for index, row in group.iterrows():
                valid, reason = self.validate_row(row)

                if valid:
                    matches.append(index)

            if len(matches) == 1:
                kept_index = matches[0]

                kept_indices.add(kept_index)

                for index in group.index:
                    if index != kept_index:
                        removed_indices.add(index)

                report_rows.append(
                    {
                        "video_path": video_path,
                        "decision": "resolved",
                        "kept_rows": 1,
                        "removed_rows": len(group) - 1,
                        "reason": "exact_path_metadata_match",
                    }
                )

            else:
                ambiguous_indices.update(
                    group.index.tolist()
                )

                report_rows.append(
                    {
                        "video_path": video_path,
                        "decision": "ambiguous",
                        "kept_rows": 0,
                        "removed_rows": 0,
                        "reason": "no_unique_path_metadata_match",
                    }
                )

        clean_dataframe = dataframe.drop(
            index=list(removed_indices)
        ).copy()

        clean_dataframe = clean_dataframe.drop_duplicates(
            subset=["video_path"],
            keep="first",
        )

        return (
            clean_dataframe,
            report_rows,
            kept_indices,
            removed_indices,
            ambiguous_indices,
        )
#change
    def validate_final_dataset(self, dataframe):
        duplicate_count = dataframe[
            "video_path"
        ].duplicated().sum()

        class_count = dataframe["label"].nunique()

        parent_count = dataframe[
            "parent_label"
        ].nunique()

        split_counts = (
            dataframe["split"]
            .value_counts()
            .sort_index()
        )

        print()
        print("FINAL DATASET VALIDATION")
        print("=" * 50)
        print(
            f"Total metadata rows: "
            f"{len(dataframe)}"
        )
        print(
            f"Unique video paths: "
            f"{dataframe['video_path'].nunique()}"
        )
        print(
            f"Duplicate video paths: "
            f"{duplicate_count}"
        )
        print(
            f"Number of classes: "
            f"{class_count}"
        )
        print(
            f"Number of parent categories: "
            f"{parent_count}"
        )

        print()
        print("Split distribution:")
        print(split_counts.to_string())

        print()

        if duplicate_count == 0:
            print("PASS: No duplicate video paths remain.")
        else:
            print(
                "WARNING: Duplicate video paths remain."
            )

        if class_count == 50:
            print("PASS: 50 classes are present.")
        else:
            print(
                f"WARNING: Expected 50 classes, "
                f"found {class_count}."
            )

        return {
            "rows": len(dataframe),
            "unique_video_paths": dataframe[
                "video_path"
            ].nunique(),
            "duplicate_video_paths": duplicate_count,
            "classes": class_count,
            "parent_categories": parent_count,
            "split_counts": split_counts.to_dict(),
        }

    def run(self):
        print("Loading metadata...")
        dataframe = self.load_metadata()

        print(
            f"Original metadata rows: "
            f"{len(dataframe)}"
        )

        print(
            f"Original unique video paths: "
            f"{dataframe['video_path'].nunique()}"
        )

        duplicate_count = dataframe[
            "video_path"
        ].duplicated().sum()

        print(
            f"Duplicate metadata rows: "
            f"{duplicate_count}"
        )

        if duplicate_count == 0:
            print()
            print(
                "No duplicate video paths found."
            )

            dataframe.to_csv(
                self.output_path,
                index=False,
            )

            self.validate_final_dataset(
                dataframe
            )

            return

        (
            clean_dataframe,
            report_rows,
            kept_indices,
            removed_indices,
            ambiguous_indices,
        ) = self.clean_duplicates(
            dataframe
        )

        report_dataframe = pd.DataFrame(
            report_rows
        )

        clean_dataframe.to_csv(
            self.output_path,
            index=False,
        )

        report_dataframe.to_csv(
            self.report_path,
            index=False,
        )

        print()
        print("CLEANING RESULTS")
        print("=" * 50)
        print(
            f"Original rows: "
            f"{len(dataframe)}"
        )
        print(
            f"Cleaned rows: "
            f"{len(clean_dataframe)}"
        )
        print(
            f"Rows removed: "
            f"{len(removed_indices)}"
        )
        print(
            f"Ambiguous rows: "
            f"{len(ambiguous_indices)}"
        )

        resolved_count = sum(
            1
            for row in report_rows
            if row["decision"] == "resolved"
        )

        deduplicated_count = sum(
            1
            for row in report_rows
            if row["decision"] == "deduplicated"
        )

        ambiguous_count = sum(
            1
            for row in report_rows
            if row["decision"] == "ambiguous"
        )

        print(
            f"Resolved duplicate paths: "
            f"{resolved_count}"
        )
        print(
            f"Identical duplicate paths: "
            f"{deduplicated_count}"
        )
        print(
            f"Ambiguous duplicate paths: "
            f"{ambiguous_count}"
        )

        print()
        print(
            f"Clean metadata saved to:"
        )
        print(self.output_path)

        print()
        print(
            f"Quality report saved to:"
        )
        print(self.report_path)

        self.validate_final_dataset(
            clean_dataframe
        )


if __name__ == "__main__":
    cleaner = MetadataCleaner()
    cleaner.run()