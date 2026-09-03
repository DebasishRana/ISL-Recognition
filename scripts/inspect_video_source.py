"""
Functionality: This script inspects the files available from the dataset's original video source.

"""

from pathlib import Path
from urllib.request import Request, urlopen
import json

import pandas as pd




RECORD_ID = "4010759"

API_URL = (
    f"https://zenodo.org/api/records/{RECORD_ID}"
)

OUTPUT_DIR = Path(
    r"D:\ISL-Dataset\metadata"
)

OUTPUT_FILE = (
    OUTPUT_DIR / "video_source_files.csv"
)



def format_size(size_bytes):
    """Convert bytes into a human-readable size."""

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



def main():

    print("=" * 60)
    print("VIDEO SOURCE INSPECTION")
    print("=" * 60)

    print()
    print("Connecting to video source...")
    print(
        f"Record ID: {RECORD_ID}"
    )
    print()



    request = Request(
        API_URL,
        headers={
            "User-Agent": (
                "ISL-Recognition-Dataset-Inspector/1.0"
            )
        },
    )

    try:

        with urlopen(
            request,
            timeout=30,
        ) as response:

            data = json.load(response)

    except Exception as error:

        raise RuntimeError(
            "Could not retrieve the video source information.\n"
            f"URL: {API_URL}\n"
            f"Error: {error}"
        )



    files = data.get(
        "files",
        []
    )

    if not files:

        raise RuntimeError(
            "No files were found in the source record."
        )

    print(
        f"Files found: {len(files)}"
    )

    print()



    file_information = []

    total_size = 0

    for index, file_info in enumerate(
        files,
        start=1,
    ):

        filename = file_info.get(
            "key",
            "Unknown"
        )

        size = file_info.get(
            "size",
            0
        )

        download_url = file_info.get(
            "links",
            {}
        ).get(
            "self"
        )

        total_size += size

        file_information.append(
            {
                "file_name": filename,
                "size_bytes": size,
                "size_readable": format_size(size),
                "download_url": download_url,
            }
        )

        print(
            f"{index:3}. "
            f"{filename}"
        )

        print(
            f"     Size: {format_size(size)}"
        )



    dataframe = pd.DataFrame(
        file_information
    )



    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )



    print()
    print("=" * 60)
    print("VIDEO SOURCE SUMMARY")
    print("=" * 60)

    print()
    print(
        "Number of files:",
        len(files)
    )

    print(
        "Total size:",
        format_size(total_size)
    )

    print()
    print(
        "CSV saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()