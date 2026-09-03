"""
Functionality: This module validates the complete extracted landmark dataset.

"""

from pathlib import Path
import csv

import numpy as np
import pandas as pd


class ExtractedDataValidator:
    def __init__(self):
        self.dataset_root = Path(
            r"D:\ISL-Dataset"
        )

        self.metadata_path = (
            self.dataset_root
            / "metadata"
            / "dataset_metadata_clean.csv"
        )

        self.sequence_root = (
            self.dataset_root
            / "extracted"
            / "sequences"
        )

        self.manifest_path = (
            self.dataset_root
            / "extracted"
            / "extraction_manifest.csv"
        )

        self.report_path = (
            self.dataset_root
            / "extracted"
            / "extraction_validation_report.csv"
        )

        self.expected_sequence_length = 30
        self.expected_feature_count = 126
        self.expected_classes = 50
        self.expected_samples = 943

    def load_metadata(self):
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: "
                f"{self.metadata_path}"
            )

        return pd.read_csv(
            self.metadata_path
        )

    def load_manifest(self):
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest file not found: "
                f"{self.manifest_path}"
            )

        return pd.read_csv(
            self.manifest_path
        )

    def validate_sequences(self):
        sequence_files = sorted(
            self.sequence_root.glob("*.npy")
        )

        results = []

        for position, sequence_file in enumerate(
            sequence_files,
            start=1,
        ):
            result = {
                "sequence_file": sequence_file.name,
                "shape_valid": False,
                "dtype_valid": False,
                "nan_free": False,
                "inf_free": False,
                "valid": False,
                "error": "",
            }

            try:
                sequence = np.load(
                    sequence_file
                )

                shape_valid = (
                    sequence.shape
                    == (
                        self.expected_sequence_length,
                        self.expected_feature_count,
                    )
                )

                dtype_valid = (
                    sequence.dtype
                    == np.float32
                )

                nan_free = not np.isnan(
                    sequence
                ).any()

                inf_free = not np.isinf(
                    sequence
                ).any()

                result["shape_valid"] = (
                    shape_valid
                )

                result["dtype_valid"] = (
                    dtype_valid
                )

                result["nan_free"] = (
                    nan_free
                )

                result["inf_free"] = (
                    inf_free
                )

                result["valid"] = all(
                    [
                        shape_valid,
                        dtype_valid,
                        nan_free,
                        inf_free,
                    ]
                )

                if not result["valid"]:
                    result["error"] = (
                        f"shape={sequence.shape}, "
                        f"dtype={sequence.dtype}"
                    )

            except Exception as error:
                result["error"] = str(
                    error
                )

            results.append(
                result
            )

            if position % 100 == 0:
                print(
                    f"Validated "
                    f"{position} sequences..."
                )

        return sequence_files, results

    def validate_metadata(self, metadata):
        duplicate_paths = (
            metadata["video_path"]
            .duplicated()
            .sum()
        )

        class_count = (
            metadata["label"]
            .nunique()
        )

        parent_count = (
            metadata["parent_label"]
            .nunique()
        )

        split_distribution = (
            metadata["split"]
            .value_counts()
            .sort_index()
        )

        return (
            duplicate_paths,
            class_count,
            parent_count,
            split_distribution,
        )

    def validate_manifest(
        self,
        metadata,
        manifest,
    ):
        metadata_paths = set(
            metadata["video_path"]
            .astype(str)
        )

        manifest_paths = set(
            manifest["video_path"]
            .astype(str)
        )

        missing_from_manifest = (
            metadata_paths
            - manifest_paths
        )

        extra_in_manifest = (
            manifest_paths
            - metadata_paths
        )

        duplicate_manifest_paths = (
            manifest["video_path"]
            .duplicated()
            .sum()
        )

        return (
            missing_from_manifest,
            extra_in_manifest,
            duplicate_manifest_paths,
        )

    def validate_sequence_count(
        self,
        sequence_files,
    ):
        sequence_count = len(
            sequence_files
        )

        return (
            sequence_count
            == self.expected_samples
        )

    def write_report(
        self,
        results,
    ):
        dataframe = pd.DataFrame(
            results
        )

        dataframe.to_csv(
            self.report_path,
            index=False,
        )

    def run(self):
        print()
        print(
            "EXTRACTED DATASET VALIDATION"
        )
        print(
            "=" * 50
        )

        metadata = self.load_metadata()
        manifest = self.load_manifest()

        print()
        print(
            "Loading extracted sequences..."
        )

        sequence_files, sequence_results = (
            self.validate_sequences()
        )

        valid_sequences = sum(
            result["valid"]
            for result in sequence_results
        )

        invalid_sequences = (
            len(sequence_results)
            - valid_sequences
        )

        print()
        print(
            "SEQUENCE VALIDATION"
        )
        print(
            "=" * 50
        )

        print(
            f"Sequence files: "
            f"{len(sequence_files)}"
        )

        print(
            f"Valid sequences: "
            f"{valid_sequences}"
        )

        print(
            f"Invalid sequences: "
            f"{invalid_sequences}"
        )

        print()
        print(
            "METADATA VALIDATION"
        )
        print(
            "=" * 50
        )

        (
            duplicate_paths,
            class_count,
            parent_count,
            split_distribution,
        ) = self.validate_metadata(
            metadata
        )

        print(
            f"Metadata rows: "
            f"{len(metadata)}"
        )

        print(
            f"Unique video paths: "
            f"{metadata['video_path'].nunique()}"
        )

        print(
            f"Duplicate video paths: "
            f"{duplicate_paths}"
        )

        print(
            f"Classes: "
            f"{class_count}"
        )

        print(
            f"Parent categories: "
            f"{parent_count}"
        )

        print()
        print(
            "Split distribution:"
        )

        for split, count in (
            split_distribution.items()
        ):
            print(
                f"{split}: {count}"
            )

        print()
        print(
            "MANIFEST VALIDATION"
        )
        print(
            "=" * 50
        )

        (
            missing_from_manifest,
            extra_in_manifest,
            duplicate_manifest_paths,
        ) = self.validate_manifest(
            metadata,
            manifest,
        )

        print(
            f"Manifest rows: "
            f"{len(manifest)}"
        )

        print(
            f"Missing from manifest: "
            f"{len(missing_from_manifest)}"
        )

        print(
            f"Extra in manifest: "
            f"{len(extra_in_manifest)}"
        )

        print(
            f"Duplicate manifest paths: "
            f"{duplicate_manifest_paths}"
        )

        count_pass = (
            self.validate_sequence_count(
                sequence_files
            )
        )

        shape_pass = (
            invalid_sequences == 0
        )

        metadata_pass = (
            duplicate_paths == 0
            and class_count
            == self.expected_classes
            and len(metadata)
            == self.expected_samples
        )

        manifest_pass = (
            len(missing_from_manifest) == 0
            and len(extra_in_manifest) == 0
            and duplicate_manifest_paths == 0
        )

        overall_pass = all(
            [
                count_pass,
                shape_pass,
                metadata_pass,
                manifest_pass,
            ]
        )

        self.write_report(
            sequence_results
        )

        print()
        print(
            "FINAL VALIDATION"
        )
        print(
            "=" * 50
        )

        print(
            f"Sequence count: "
            f"{'PASS' if count_pass else 'FAIL'}"
        )

        print(
            f"Sequence integrity: "
            f"{'PASS' if shape_pass else 'FAIL'}"
        )

        print(
            f"Metadata integrity: "
            f"{'PASS' if metadata_pass else 'FAIL'}"
        )

        print(
            f"Manifest integrity: "
            f"{'PASS' if manifest_pass else 'FAIL'}"
        )

        print()

        if overall_pass:
            print(
                "PASS: Extracted dataset is ready for training."
            )
        else:
            print(
                "FAIL: Dataset requires investigation before training."
            )

        print()
        print(
            "Validation report saved to:"
        )
        print(
            self.report_path
        )


if __name__ == "__main__":
    validator = (
        ExtractedDataValidator()
    )

    validator.run()