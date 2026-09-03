"""
Functionality: This module extracts fixed-length hand landmark sequences from the ISL dataset videos.

"""

from pathlib import Path
import argparse
import csv
import shutil
import tempfile
import zipfile

import cv2
import numpy as np
import pandas as pd

from hand_tracker import HandTracker
from feature_extractor import HandFeatureExtractor


class LandmarkExtractor:
    def __init__(self):
        self.dataset_root = Path(r"D:\ISL-Dataset")

        self.metadata_path = (
            self.dataset_root
            / "metadata"
            / "dataset_metadata_clean.csv"
        )

        self.mapping_path = (
            self.dataset_root
            / "metadata"
            / "archive_video_mapping.csv"
        )

        self.video_root = (
            self.dataset_root
            / "videos"
        )

        self.output_root = (
            self.dataset_root
            / "extracted"
        )

        self.sequence_root = (
            self.output_root
            / "sequences"
        )

        self.manifest_path = (
            self.output_root
            / "extraction_manifest.csv"
        )

        self.failed_path = (
            self.output_root
            / "failed_videos.csv"
        )

        self.sequence_length = 30
        self.feature_count = 126

        self.sequence_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    def load_metadata(self):
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Clean metadata not found at: "
                f"{self.metadata_path}"
            )

        return pd.read_csv(
            self.metadata_path
        )

    def find_mapping_columns(self, dataframe):
        archive_candidates = [
            "archive_file",
            "archive",
            "zip_file",
            "filename",
            "file_name",
        ]

        video_candidates = [
            "video_path",
            "path",
        ]

        archive_column = None
        video_column = None

        for column in archive_candidates:
            if column in dataframe.columns:
                archive_column = column
                break

        for column in video_candidates:
            if column in dataframe.columns:
                video_column = column
                break

        if archive_column is None:
            raise ValueError(
                "Could not find archive filename column."
            )

        if video_column is None:
            raise ValueError(
                "Could not find video path column."
            )

        return archive_column, video_column

    def load_mapping(self):
        if not self.mapping_path.exists():
            raise FileNotFoundError(
                f"Archive mapping not found at: "
                f"{self.mapping_path}"
            )

        dataframe = pd.read_csv(
            self.mapping_path
        )

        archive_column, video_column = (
            self.find_mapping_columns(dataframe)
        )

        mapping = {}

        for _, row in dataframe.iterrows():
            video_path = str(
                row[video_column]
            ).replace(
                "\\",
                "/",
            )

            archive_file = str(
                row[archive_column]
            )

            mapping[video_path] = archive_file

        return mapping

    def locate_archive(self, archive_file):
        direct_path = (
            self.video_root
            / archive_file
        )

        if direct_path.exists():
            return direct_path

        matches = list(
            self.video_root.rglob(
                archive_file
            )
        )

        if matches:
            return matches[0]

        return None

    def extract_video_from_zip(
        self,
        archive_path,
        video_path,
        temporary_directory,
    ):
        video_path = str(
            video_path
        ).replace(
            "\\",
            "/",
        )

        with zipfile.ZipFile(
            archive_path,
            "r",
        ) as archive:

            matching_member = None

            for member in archive.namelist():
                normalized_member = member.replace(
                    "\\",
                    "/",
                )

                if normalized_member == video_path:
                    matching_member = member
                    break

            if matching_member is None:
                filename_matches = [
                    member
                    for member in archive.namelist()
                    if Path(member).name
                    == Path(video_path).name
                ]

                if len(filename_matches) == 1:
                    matching_member = (
                        filename_matches[0]
                    )

            if matching_member is None:
                raise FileNotFoundError(
                    f"Video not found inside archive: "
                    f"{video_path}"
                )

            extracted_path = (
                Path(temporary_directory)
                / Path(video_path).name
            )

            with archive.open(
                matching_member
            ) as source:
                with open(
                    extracted_path,
                    "wb",
                ) as destination:
                    shutil.copyfileobj(
                        source,
                        destination,
                    )

        return extracted_path

    def read_video_frames(self, video_path):
        capture = cv2.VideoCapture(
            str(video_path)
        )

        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open video: "
                f"{video_path}"
            )

        frames = []

        while True:
            success, frame = capture.read()

            if not success:
                break

            frames.append(frame)

        capture.release()

        if not frames:
            raise RuntimeError(
                "Video contains no readable frames."
            )

        return frames

    def select_frames(self, frames):
        total_frames = len(frames)

        if total_frames >= self.sequence_length:
            indices = np.linspace(
                0,
                total_frames - 1,
                self.sequence_length,
            ).astype(int)

            return [
                frames[index]
                for index in indices
            ]

        selected = list(frames)

        while len(selected) < self.sequence_length:
            selected.append(
                frames[-1]
            )

        return selected

    def extract_sequence(
        self,
        video_path,
    ):
        frames = self.read_video_frames(
            video_path
        )

        selected_frames = self.select_frames(
            frames
        )

        tracker = HandTracker()
        extractor = HandFeatureExtractor()

        sequence = []

        try:
            for frame_index, frame in enumerate(
                selected_frames
            ):
                timestamp_ms = (
                    frame_index * 33
                )

                result = tracker.process(
                    frame,
                    timestamp_ms,
                )

                features = extractor.extract(
                    result
                )

                sequence.append(features)

        finally:
            tracker.close()

        sequence = np.asarray(
            sequence,
            dtype=np.float32,
        )

        if sequence.shape != (
            self.sequence_length,
            self.feature_count,
        ):
            raise RuntimeError(
                f"Unexpected sequence shape: "
                f"{sequence.shape}"
            )

        return sequence

    def create_output_name(
        self,
        row_index,
        video_path,
    ):
        filename = Path(
            video_path
        ).stem

        safe_filename = "".join(
            character
            if character.isalnum()
            else "_"
            for character in filename
        )

        return (
            f"{row_index:05d}_"
            f"{safe_filename}.npy"
        )

    def write_manifest(
        self,
        rows,
    ):
        fieldnames = [
            "sequence_file",
            "video_path",
            "parent_label",
            "label",
            "split",
            "status",
        ]

        with open(
            self.manifest_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(rows)

    def write_failed(
        self,
        rows,
    ):
        fieldnames = [
            "video_path",
            "parent_label",
            "label",
            "split",
            "error",
        ]

        with open(
            self.failed_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(rows)

    def run(self, limit=None):
        metadata = self.load_metadata()
        mapping = self.load_mapping()

        if limit is not None:
            metadata = metadata.head(
                limit
            ).copy()

        successful = []
        failed = []

        total = len(metadata)

        print()
        print("LANDMARK EXTRACTION")
        print("=" * 50)
        print(
            f"Videos to process: {total}"
        )
        print(
            f"Sequence length: "
            f"{self.sequence_length}"
        )
        print(
            f"Features per frame: "
            f"{self.feature_count}"
        )
        print(
            "Resume mode: enabled"
        )

        for position, (
            index,
            row,
        ) in enumerate(
            metadata.iterrows(),
            start=1,
        ):
            video_path = str(
                row["video_path"]
            ).replace(
                "\\",
                "/",
            )

            output_name = (
                self.create_output_name(
                    index,
                    video_path,
                )
            )

            output_path = (
                self.sequence_root
                / output_name
            )

            print()
            print(
                f"[{position}/{total}] "
                f"{video_path}"
            )

            if output_path.exists():
                try:
                    existing_sequence = (
                        np.load(
                            output_path
                        )
                    )

                    if existing_sequence.shape == (
                        self.sequence_length,
                        self.feature_count,
                    ):
                        print(
                            "SKIPPED: "
                            "sequence already exists."
                        )

                        successful.append(
                            {
                                "sequence_file": output_name,
                                "video_path": video_path,
                                "parent_label": row[
                                    "parent_label"
                                ],
                                "label": row["label"],
                                "split": row["split"],
                                "status": "success",
                            }
                        )

                        continue

                except Exception:
                    output_path.unlink(
                        missing_ok=True
                    )

            archive_file = mapping.get(
                video_path
            )

            if archive_file is None:
                error = (
                    "No archive mapping found."
                )

                print(
                    f"FAILED: {error}"
                )

                failed.append(
                    {
                        "video_path": video_path,
                        "parent_label": row[
                            "parent_label"
                        ],
                        "label": row["label"],
                        "split": row["split"],
                        "error": error,
                    }
                )

                continue

            archive_path = self.locate_archive(
                archive_file
            )

            if archive_path is None:
                error = (
                    f"Archive not found: "
                    f"{archive_file}"
                )

                print(
                    f"FAILED: {error}"
                )

                failed.append(
                    {
                        "video_path": video_path,
                        "parent_label": row[
                            "parent_label"
                        ],
                        "label": row["label"],
                        "split": row["split"],
                        "error": error,
                    }
                )

                continue

            temporary_directory = tempfile.mkdtemp(
                prefix="isl_extract_"
            )

            try:
                extracted_video = (
                    self.extract_video_from_zip(
                        archive_path,
                        video_path,
                        temporary_directory,
                    )
                )

                sequence = (
                    self.extract_sequence(
                        extracted_video
                    )
                )

                np.save(
                    output_path,
                    sequence,
                )

                successful.append(
                    {
                        "sequence_file": output_name,
                        "video_path": video_path,
                        "parent_label": row[
                            "parent_label"
                        ],
                        "label": row["label"],
                        "split": row["split"],
                        "status": "success",
                    }
                )

                print(
                    f"SUCCESS: "
                    f"{sequence.shape}"
                )

            except Exception as error:
                print(
                    f"FAILED: {error}"
                )

                failed.append(
                    {
                        "video_path": video_path,
                        "parent_label": row[
                            "parent_label"
                        ],
                        "label": row["label"],
                        "split": row["split"],
                        "error": str(error),
                    }
                )

            finally:
                shutil.rmtree(
                    temporary_directory,
                    ignore_errors=True,
                )

        self.write_manifest(
            successful
        )

        self.write_failed(
            failed
        )

        print()
        print("EXTRACTION COMPLETE")
        print("=" * 50)
        print(
            f"Successful: "
            f"{len(successful)}"
        )
        print(
            f"Failed: "
            f"{len(failed)}"
        )

        print()
        print(
            "Sequences saved to:"
        )
        print(
            self.sequence_root
        )

        print()
        print(
            "Manifest saved to:"
        )
        print(
            self.manifest_path
        )

        print()
        print(
            "Failed-video report saved to:"
        )
        print(
            self.failed_path
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    arguments = parser.parse_args()

    extractor = LandmarkExtractor()

    extractor.run(
        limit=arguments.limit
    )