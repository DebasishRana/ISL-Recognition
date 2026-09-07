"""
Functionality: This module provides logging for successfully recognized signs.

"""

from pathlib import Path
from datetime import datetime
import csv


class SignLogger:
    """
    Handles persistent logging of accepted sign recognition events.
    """

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[1]

        self.log_directory = (
            self.project_root
            / "logs"
        )

        self.log_file = (
            self.log_directory
            / "recognized_signs.csv"
        )

        self.log_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize_log_file()

    def _initialize_log_file(self):
        if self.log_file.exists():
            return

        with open(
            self.log_file,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    "timestamp",
                    "sign",
                    "confidence",
                    "duration_seconds",
                    "frame_count",
                    "status",
                ]
            )

    def log_sign(
        self,
        sign,
        confidence,
        duration_seconds,
        frame_count,
        status="ACCEPTED",
    ):
        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        with open(
            self.log_file,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    timestamp,
                    sign,
                    f"{confidence:.4f}",
                    f"{duration_seconds:.2f}",
                    frame_count,
                    status,
                ]
            )

        return self.log_file