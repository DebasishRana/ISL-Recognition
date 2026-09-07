"""
Functionality: This module provides local text-to-speech functionality using Piper.

"""

from pathlib import Path
import subprocess
import tempfile
import sys

import winsound


class TTSEngine:

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[1]

        self.voice_directory = (
            self.project_root
            / "models"
            / "tts"
        )

        self.voice_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.voices = {}
        self.current_voice = None
        self.current_process = None

        self._discover_voices()

        if not self.voices:
            raise FileNotFoundError(
                "No Piper voices found in "
                f"{self.voice_directory}"
            )

        self.current_voice = next(
            iter(self.voices)
        )

    def _discover_voices(self):

        self.voices.clear()

        for model_path in self.voice_directory.rglob(
            "*.onnx"
        ):

            config_path = Path(
                f"{model_path}.json"
            )

            if not config_path.exists():
                continue

            voice_name = model_path.stem

            self.voices[voice_name] = {
                "model": model_path,
                "config": config_path,
            }

    def get_voices(self):

        return list(
            self.voices.keys()
        )

    def get_current_voice(self):

        return self.current_voice

    def set_voice(self, voice_name):

        if voice_name not in self.voices:
            raise ValueError(
                f"Voice not found: {voice_name}"
            )

        self.current_voice = voice_name

    def speak(self, text):

        if not text:
            return

        text = str(text).strip()

        if not text:
            return

        if self.current_voice is None:
            raise RuntimeError(
                "No Piper voice selected."
            )

        self.stop()

        voice = self.voices[
            self.current_voice
        ]

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as temp_file:

            output_path = Path(
                temp_file.name
            )

        command = [
            sys.executable,
            "-m",
            "piper",
            "-m",
            str(voice["model"]),
            "-f",
            str(output_path),
            "--",
            text,
        ]

        try:

            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            self.current_process = output_path

            winsound.PlaySound(
                str(output_path),
                winsound.SND_FILENAME,
            )

        finally:

            if output_path.exists():
                output_path.unlink()

            self.current_process = None

    def stop(self):

        winsound.PlaySound(
            None,
            winsound.SND_PURGE,
        )

        if (
            self.current_process is not None
            and self.current_process.exists()
        ):
            self.current_process.unlink()

        self.current_process = None

    def close(self):

        self.stop()