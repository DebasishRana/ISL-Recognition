"""
Functionality:
This module extracts multimodal feature sequences from
the ISL video dataset.

Processing pipeline:

Video archive
    ↓
Temporary video extraction
    ↓
MultimodalTracker
    ↓
Hands + Pose + Face landmarks
    ↓
FeatureExtractor
    ↓
1662 features per frame
    ↓
30-frame sequence
    ↓
.npy file

The same feature extractor is used for every video.
Missing modalities are handled by the feature extractor.
"""

from pathlib import Path
import tempfile
import zipfile

import cv2
import numpy as np
import pandas as pd

from multimodal_tracker import MultimodalTracker
from feature_extractor import FeatureExtractor


class LandmarkExtractor:

    def __init__(self):
        self.dataset_root = Path(
            r"D:\ISL-Dataset"
        )

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
            / "multimodal_sequences"
        )

        self.manifest_path = (
            self.output_root
            / "multimodal_extraction_manifest.csv"
        )

        self.failed_path = (
            self.output_root
            / "multimodal_failed_videos.csv"
        )

        self.sequence_length = 30

        self.feature_extractor = FeatureExtractor()

        self.feature_count = (
            self.feature_extractor.total_features
        )

        self.output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.sequence_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    def load_metadata(self):

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found: "
                f"{self.metadata_path}"
            )

        return pd.read_csv(
            self.metadata_path
        )
    #chnage
    def load_mapping(self):

        if not self.mapping_path.exists():
            raise FileNotFoundError(
                f"Archive mapping not found: "
                f"{self.mapping_path}"
            )

        mapping_df = pd.read_csv(
            self.mapping_path
        )

        required_columns = {
            "video_path",
            "archive",
        }

        missing_columns = (
            required_columns
            - set(mapping_df.columns)
        )

        if missing_columns:
            raise ValueError(
                "Missing required columns in "
                f"archive mapping: {missing_columns}"
            )

        mapping = {}

        for _, row in mapping_df.iterrows():

            video_path = str(
                row["video_path"]
            ).replace(
                "\\",
                "/",
            ).strip()

            archive_file = str(
                row["archive"]
            ).strip()

            if video_path and archive_file:
                mapping[video_path] = archive_file

        return mapping
#change
    def locate_archive(
        self,
        archive_file,
    ):

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

        normalized_video = (
            str(video_path)
            .replace(
                "\\",
                "/",
            )
            .strip("/")
        )

        with zipfile.ZipFile(
            archive_path,
            "r",
        ) as archive:

            members = archive.namelist()

            exact_match = None

            for member in members:

                normalized_member = (
                    member
                    .replace(
                        "\\",
                        "/",
                    )
                    .strip("/")
                )

                if (
                    normalized_member
                    == normalized_video
                ):
                    exact_match = member
                    break

            if exact_match is None:

                target_name = Path(
                    normalized_video
                ).name

                basename_matches = [
                    member
                    for member in members
                    if Path(
                        member
                    ).name.lower()
                    == target_name.lower()
                ]

                if len(
                    basename_matches
                ) == 1:

                    exact_match = (
                        basename_matches[0]
                    )

            if exact_match is None:
                raise FileNotFoundError(
                    f"Video not found in archive: "
                    f"{video_path}"
                )

            output_path = (
                Path(temporary_directory)
                / Path(exact_match).name
            )

            with archive.open(
                exact_match
            ) as source:

                with open(
                    output_path,
                    "wb",
                ) as destination:

                    destination.write(
                        source.read()
                    )

        return output_path

    def read_video_frames(
        self,
        video_path,
    ):

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

            success, frame = (
                capture.read()
            )

            if not success:
                break

            frames.append(frame)

        capture.release()

        if not frames:
            raise RuntimeError(
                f"No frames found: "
                f"{video_path}"
            )

        return frames

    def select_frames(
        self,
        frames,
    ):

        if len(frames) == 0:
            raise ValueError(
                "Video contains no frames."
            )

        indices = np.linspace(
            0,
            len(frames) - 1,
            self.sequence_length,
        ).astype(int)

        return [
            frames[index]
            for index in indices
        ]

    def create_output_name(
        self,
        video_path,
        index,
    ):

        stem = Path(
            video_path
        ).stem

        return (
            f"{index:05d}_{stem}.npy"
        )

    def extract_sequence(
        self,
        video_path,
    ):

        frames = self.read_video_frames(
            video_path
        )

        selected_frames = (
            self.select_frames(
                frames
            )
        )

        tracker = None

        try:

            tracker = MultimodalTracker()

            sequence = []

            for frame_index, frame in enumerate(
                selected_frames
            ):

                timestamp_ms = int(
                    frame_index * 33.333
                )

                detection_result = (
                    tracker.process(
                        frame,
                        timestamp_ms,
                    )
                )

                features = (
                    self.feature_extractor.extract(
                        detection_result
                    )
                )

                features = np.asarray(
                    features,
                    dtype=np.float32,
                )

                expected_shape = (
                    self.feature_count,
                )

                if features.shape != expected_shape:
                    raise ValueError(
                        f"Unexpected feature shape: "
                        f"{features.shape}; "
                        f"expected "
                        f"{expected_shape}"
                    )

                if not np.isfinite(
                    features
                ).all():
                    raise ValueError(
                        "Feature vector contains "
                        "NaN or Inf values."
                    )

                sequence.append(
                    features
                )

        finally:

            if tracker is not None:
                tracker.close()

        sequence = np.asarray(
            sequence,
            dtype=np.float32,
        )

        expected_sequence_shape = (
            self.sequence_length,
            self.feature_count,
        )

        if (
            sequence.shape
            != expected_sequence_shape
        ):
            raise ValueError(
                f"Unexpected sequence shape: "
                f"{sequence.shape}; "
                f"expected "
                f"{expected_sequence_shape}"
            )

        return sequence

    def run(
        self,
        limit=None,
    ):

        metadata = self.load_metadata()

        mapping = self.load_mapping()

        if limit is not None:
            metadata = metadata.head(
                limit
            )

        manifest_rows = []

        failed_rows = []

        total = len(metadata)

        for position, (_, row) in enumerate(
            metadata.iterrows(),
            start=1,
        ):

            video_path = str(
                row["video_path"]
            ).replace(
                "\\",
                "/",
            )

            parent_label = str(
                row["parent_label"]
            )

            label = str(
                row["label"]
            )

            split = str(
                row["split"]
            )

            output_name = (
                self.create_output_name(
                    video_path,
                    position - 1,
                )
            )

            output_path = (
                self.sequence_root
                / output_name
            )

            try:

                archive_file = mapping.get(
                    video_path
                )

                if archive_file is None:
                    raise FileNotFoundError(
                        "Archive mapping not found."
                    )

                archive_path = (
                    self.locate_archive(
                        archive_file
                    )
                )

                if archive_path is None:
                    raise FileNotFoundError(
                        f"Archive not found: "
                        f"{archive_file}"
                    )

                with tempfile.TemporaryDirectory() as temp_dir:

                    temporary_video = (
                        self.extract_video_from_zip(
                            archive_path,
                            video_path,
                            temp_dir,
                        )
                    )

                    sequence = (
                        self.extract_sequence(
                            temporary_video
                        )
                    )

                np.save(
                    output_path,
                    sequence,
                )

                manifest_rows.append(
                    {
                        "sequence_file": output_name,
                        "video_path": video_path,
                        "parent_label": parent_label,
                        "label": label,
                        "split": split,
                        "status": "success",
                    }
                )

                print(
                    f"[{position}/{total}] "
                    f"SUCCESS {label}"
                )

            except Exception as error:

                error_message = (
                    f"{type(error).__name__}: "
                    f"{error}"
                )

                manifest_rows.append(
                    {
                        "sequence_file": output_name,
                        "video_path": video_path,
                        "parent_label": parent_label,
                        "label": label,
                        "split": split,
                        "status": "failed",
                    }
                )

                failed_rows.append(
                    {
                        "video_path": video_path,
                        "label": label,
                        "error": error_message,
                    }
                )

                print(
                    f"[{position}/{total}] "
                    f"FAILED {label}"
                )

                print(
                    f"    {error_message}"
                )

        pd.DataFrame(
            manifest_rows
        ).to_csv(
            self.manifest_path,
            index=False,
        )

        pd.DataFrame(
            failed_rows
        ).to_csv(
            self.failed_path,
            index=False,
        )

        successful = (
            len(manifest_rows)
            - len(failed_rows)
        )

        print()
        print("MULTIMODAL EXTRACTION SUMMARY")
        print("==============================")
        print(
            f"Videos processed : {total}"
        )
        print(
            f"Successful       : {successful}"
        )
        print(
            f"Failed           : {len(failed_rows)}"
        )
        print(
            f"Features/frame   : {self.feature_count}"
        )
        print(
            f"Sequence shape   : "
            f"(30, {self.feature_count})"
        )
        print(
            f"Output directory : "
            f"{self.sequence_root}"
        )
        print(
            f"Manifest         : "
            f"{self.manifest_path}"
        )


if __name__ == "__main__":

    extractor = LandmarkExtractor()

    extractor.run()