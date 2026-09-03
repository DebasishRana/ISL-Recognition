"""
Functionality: This script downloads the required video archives for the ISL recognition project.

"""

from pathlib import Path
import time

import pandas as pd
import requests


DATASET_ROOT = Path("D:/ISL-Dataset")
METADATA_ROOT = DATASET_ROOT / "metadata"
VIDEO_ROOT = DATASET_ROOT / "videos"

SELECTED_FILE = METADATA_ROOT / "required_video_files.csv"

API_URL = "https://zenodo.org/api/records/4010759"

CHUNK_SIZE = 1024 * 1024
TIMEOUT = 60
MAX_RETRIES = 10
RETRY_DELAY = 10


def load_selected_files():
    dataframe = pd.read_csv(SELECTED_FILE)

    if "file_name" not in dataframe.columns:
        raise KeyError(
            f"Column 'file_name' not found.\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    return dataframe["file_name"].astype(str).tolist()


def load_source_files():
    response = requests.get(
        API_URL,
        timeout=TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    files = {}

    for item in data.get("files", []):
        key = item.get("key")
        links = item.get("links", {})
        download_url = links.get("self")
        size = item.get("size")

        if key and download_url and size is not None:
            files[key] = {
                "url": download_url,
                "size": int(size),
            }

    return files


def download_file(filename, source_info):
    VIDEO_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = VIDEO_ROOT / filename
    expected_size = source_info["size"]
    download_url = source_info["url"]

    if destination.exists():
        existing_size = destination.stat().st_size

        if existing_size == expected_size:
            print(f"SKIP: {filename}")
            print("      Already complete and verified.")
            return True

        if existing_size > expected_size:
            destination.unlink()
            existing_size = 0
    else:
        existing_size = 0

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            headers = {}

            if existing_size > 0:
                headers["Range"] = f"bytes={existing_size}-"

            print()
            print(f"DOWNLOAD: {filename}")
            print(f"Attempt  : {attempt}/{MAX_RETRIES}")
            print(f"Expected : {expected_size:,} bytes")
            print(f"Existing : {existing_size:,} bytes")

            with requests.get(
                download_url,
                headers=headers,
                stream=True,
                timeout=TIMEOUT,
            ) as response:

                if existing_size > 0 and response.status_code == 200:
                    existing_size = 0
                    destination.unlink(missing_ok=True)

                response.raise_for_status()

                mode = "ab" if existing_size > 0 else "wb"

                downloaded = existing_size
                start_time = time.time()

                with open(destination, mode) as file:
                    for chunk in response.iter_content(
                        chunk_size=CHUNK_SIZE
                    ):
                        if not chunk:
                            continue

                        file.write(chunk)
                        downloaded += len(chunk)

                        elapsed = max(
                            time.time() - start_time,
                            0.001,
                        )

                        speed = (
                            downloaded
                            / elapsed
                            / (1024 * 1024)
                        )

                        progress = (
                            downloaded
                            / expected_size
                            * 100
                        )

                        print(
                            f"\rProgress: {progress:6.2f}% | "
                            f"{downloaded / (1024 * 1024):,.1f} MB | "
                            f"{speed:,.2f} MB/s",
                            end="",
                            flush=True,
                        )

            print()

            final_size = destination.stat().st_size

            if final_size == expected_size:
                print("Verified: size matches source.")
                return True

            print()
            print("ERROR: Size mismatch.")
            print(f"Expected: {expected_size:,}")
            print(f"Actual  : {final_size:,}")

            existing_size = final_size

        except requests.RequestException as error:
            print()
            print("DOWNLOAD ERROR:")
            print(error)

            if destination.exists():
                existing_size = destination.stat().st_size
            else:
                existing_size = 0

        except OSError as error:
            print()
            print("FILE ERROR:")
            print(error)

            if destination.exists():
                existing_size = destination.stat().st_size
            else:
                existing_size = 0

        if attempt < MAX_RETRIES:
            print()
            print(
                f"Retrying in {RETRY_DELAY} seconds..."
            )
            time.sleep(RETRY_DELAY)

    print()
    print(f"FAILED: {filename}")
    print("Maximum retry attempts reached.")

    return False


def main():
    selected_files = load_selected_files()
    source_files = load_source_files()

    missing = [
        filename
        for filename in selected_files
        if filename not in source_files
    ]

    if missing:
        print("ERROR: Selected files not found in source:")
        print()

        for filename in missing:
            print(filename)

        return

    total_size = sum(
        source_files[filename]["size"]
        for filename in selected_files
    )

    print()
    print("ISL DATASET DOWNLOADER")
    print("=" * 60)
    print(f"Archives total : {len(selected_files)}")
    print(
        f"Total size     : "
        f"{total_size / (1024 ** 3):.2f} GB"
    )
    print(f"Download folder: {VIDEO_ROOT}")
    print()
    print("The downloader will process every archive")
    print("automatically, one by one.")
    print()
    print("Press Ctrl+C to stop safely.")
    print()

    completed = 0

    for index, filename in enumerate(
        selected_files,
        start=1,
    ):
        print()
        print("=" * 60)
        print(
            f"ARCHIVE {index}/{len(selected_files)}"
        )
        print("=" * 60)

        success = download_file(
            filename,
            source_files[filename],
        )

        if not success:
            print()
            print(
                "The downloader stopped because an archive "
                "could not be completed."
            )
            print()
            print(
                "Run the script again later to resume."
            )
            return

        completed += 1

    print()
    print("=" * 60)
    print("ALL DATASET ARCHIVES DOWNLOADED")
    print("=" * 60)
    print(
        f"Successfully processed: "
        f"{completed}/{len(selected_files)}"
    )
    print(
        f"Location: {VIDEO_ROOT}"
    )


if __name__ == "__main__":
    main()