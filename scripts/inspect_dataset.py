"""
Functionality: This script inspects the ISL dataset metadata.

"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset


SOURCE_DATASET = "ai4bharat/INCLUDE"

OUTPUT_DIR = Path(
    r"D:\ISL-Dataset\metadata"
)

OUTPUT_FILE = (
    OUTPUT_DIR / "dataset_metadata.csv"
)




def main():

    print("=" * 60)
    print("ISL DATASET INSPECTION")
    print("=" * 60)

    print()
    print("Loading dataset metadata...")
    print("This may take a little while the first time.")
    print()

    dataset = load_dataset(
        SOURCE_DATASET
    )

    all_data = []



    for split_name in dataset.keys():

        print(
            f"Processing split: {split_name}"
        )

        dataframe = dataset[
            split_name
        ].to_pandas()

        dataframe["split"] = split_name

        all_data.append(
            dataframe
        )



    metadata = pd.concat(
        all_data,
        ignore_index=True
    )

    print()
    print(
        "Total metadata rows:",
        len(metadata)
    )



    if "include_50" not in metadata.columns:

        raise RuntimeError(
            "The dataset metadata does not contain "
            "the required subset field."
        )



    isl_data = metadata[
        metadata["include_50"] == True
    ].copy()



    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )



    isl_data.to_csv(
        OUTPUT_FILE,
        index=False
    )



    print()
    print("=" * 60)
    print("ISL DATASET INFORMATION")
    print("=" * 60)

    print()
    print(
        "Total samples:",
        len(isl_data)
    )



    if "label" in isl_data.columns:

        unique_labels = sorted(
            isl_data["label"]
            .dropna()
            .unique()
        )

        print(
            "Number of sign classes:",
            len(unique_labels)
        )



    if "split" in isl_data.columns:

        print()
        print("Samples by split:")

        print(
            isl_data["split"]
            .value_counts()
            .sort_index()
        )


    if "label" in isl_data.columns:

        print()
        print("=" * 60)
        print("SAMPLES PER SIGN")
        print("=" * 60)

        class_counts = (
            isl_data["label"]
            .value_counts()
            .sort_index()
        )

        print(
            class_counts.to_string()
        )



    print()
    print("=" * 60)
    print("AVAILABLE DATA FIELDS")
    print("=" * 60)

    for column in isl_data.columns:

        print(
            f"- {column}"
        )



    if "video_path" in isl_data.columns:

        print()
        print("=" * 60)
        print("EXAMPLE VIDEO PATHS")
        print("=" * 60)

        examples = (
            isl_data["video_path"]
            .dropna()
            .head(10)
        )

        for path in examples:

            print(
                f"- {path}"
            )



    print()
    print("=" * 60)
    print("DATASET INSPECTION COMPLETE")
    print("=" * 60)

    print()
    print("Metadata saved to:")
    print(OUTPUT_FILE)
    print()


if __name__ == "__main__":
    main()