"""
Functionality: This module prepares the extracted ISL landmark sequences for machine learning.

"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


DATASET_ROOT = Path(r"D:\ISL-Dataset")
METADATA_PATH = DATASET_ROOT / "metadata" / "dataset_metadata_clean.csv"
MANIFEST_PATH = DATASET_ROOT / "extracted" / "multimodal_extraction_manifest.csv"
SEQUENCE_ROOT = DATASET_ROOT / "extracted" / "multimodal_sequences"
OUTPUT_ROOT = DATASET_ROOT / "extracted" / "processed_multimodal"

EXPECTED_SHAPE = (30, 1662)


def main():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}"
        )

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    if not SEQUENCE_ROOT.exists():
        raise FileNotFoundError(
            f"Sequence directory not found: {SEQUENCE_ROOT}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = pd.read_csv(METADATA_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)

    required_metadata_columns = {
        "video_path",
        "label",
        "split",
    }

    required_manifest_columns = {
        "sequence_file",
        "video_path",
        "label",
        "split",
        "status",
    }

    missing_metadata = (
        required_metadata_columns - set(metadata.columns)
    )

    missing_manifest = (
        required_manifest_columns - set(manifest.columns)
    )

    if missing_metadata:
        raise ValueError(
            f"Missing metadata columns: {sorted(missing_metadata)}"
        )

    if missing_manifest:
        raise ValueError(
            f"Missing manifest columns: {sorted(missing_manifest)}"
        )

    if metadata["video_path"].duplicated().any():
        raise ValueError(
            "Duplicate video paths found in cleaned metadata."
        )

    successful_manifest = manifest[
        manifest["status"] == "success"
    ].copy()

    if len(successful_manifest) != len(manifest):
        raise ValueError(
            "Manifest contains unsuccessful extraction entries."
        )

    merged = metadata.merge(
        successful_manifest[
            [
                "sequence_file",
                "video_path",
            ]
        ],
        on="video_path",
        how="left",
        validate="one_to_one",
    )

    missing_sequences = merged[
        merged["sequence_file"].isna()
    ]

    if not missing_sequences.empty:
        raise RuntimeError(
            f"Missing sequence mappings: {len(missing_sequences)}"
        )

    print("Loading extracted sequences...")

    sequences = []
    labels = []
    splits = []

    for _, row in merged.iterrows():
        sequence_path = (
            SEQUENCE_ROOT / row["sequence_file"]
        )

        if not sequence_path.exists():
            raise FileNotFoundError(
                f"Sequence file not found: {sequence_path}"
            )

        sequence = np.load(sequence_path)

        if sequence.shape != EXPECTED_SHAPE:
            raise ValueError(
                f"Invalid sequence shape for "
                f"{row['sequence_file']}: {sequence.shape}"
            )

        sequences.append(
            sequence.astype(np.float32)
        )

        labels.append(row["label"])
        splits.append(row["split"])

    X = np.asarray(
        sequences,
        dtype=np.float32,
    )

    labels = np.asarray(labels)
    splits = np.asarray(splits)

    print(f"Loaded sequences: {len(X)}")
    print(f"Sequence shape: {X.shape}")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    classes = label_encoder.classes_.tolist()

    if len(classes) != 50:
        raise ValueError(
            f"Expected 50 classes, found {len(classes)}"
        )

    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_val = X[val_mask]
    y_val = y[val_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    print()
    print("DATASET SPLIT")
    print(f"Train: {len(X_train)}")
    print(f"Validation: {len(X_val)}")
    print(f"Test: {len(X_test)}")

    print()
    print("FINAL SHAPES")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}")
    print(f"y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_test:  {y_test.shape}")

    np.save(
        OUTPUT_ROOT / "X_train.npy",
        X_train,
    )

    np.save(
        OUTPUT_ROOT / "y_train.npy",
        y_train,
    )

    np.save(
        OUTPUT_ROOT / "X_val.npy",
        X_val,
    )

    np.save(
        OUTPUT_ROOT / "y_val.npy",
        y_val,
    )

    np.save(
        OUTPUT_ROOT / "X_test.npy",
        X_test,
    )

    np.save(
        OUTPUT_ROOT / "y_test.npy",
        y_test,
    )

    with open(
        OUTPUT_ROOT / "classes.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            classes,
            file,
            indent=4,
            ensure_ascii=False,
        )

    class_mapping = pd.DataFrame(
        {
            "class_id": range(len(classes)),
            "label": classes,
        }
    )

    class_mapping.to_csv(
        OUTPUT_ROOT / "class_mapping.csv",
        index=False,
    )

    print()
    print(f"Classes: {len(classes)}")
    print(f"Processed data saved to: {OUTPUT_ROOT}")
    print("Dataset preparation completed successfully.")


if __name__ == "__main__":
    main()