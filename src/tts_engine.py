"""
Functionality:
-------------
This module provides asynchronous local text-to-speech
using Piper.

The TTSEngine class:
1. Discovers available Piper voices.
2. Allows voice selection.
3. Accepts speech requests without blocking recognition.
4. Runs Piper in a background worker.
5. Plays generated speech independently.
6. Prevents TTS from freezing the recognition loop.
7. Stops active playback when requested.
"""

from pathlib import Path
import queue
import subprocess
import tempfile
import sys
import threading

import winsound


class TTSEngine:

    def __init__(self):

        self.project_root = (
            Path(__file__).resolve().parents[1]
        )

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
        self.current_output = None

        self.speech_queue = queue.Queue(
            maxsize=1
        )

        self.stop_event = threading.Event()

        self.worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
        )

        self._discover_voices()

        if not self.voices:
            raise FileNotFoundError(
                "No Piper voices found in "
                f"{self.voice_directory}"
            )

        self.current_voice = next(
            iter(self.voices)
        )

        self.worker.start()

    def _discover_voices(self):

        self.voices.clear()

        for model_path in (
            self.voice_directory.rglob(
                "*.onnx"
            )
        ):

            config_path = Path(
                f"{model_path}.json"
            )

            if not config_path.exists():
                continue

            voice_name = (
                model_path.stem
            )

            self.voices[
                voice_name
            ] = {
                "model": model_path,
                "config": config_path,
            }

    def get_voices(self):

        return list(
            self.voices.keys()
        )

    def get_current_voice(self):

        return self.current_voice

    def set_voice(
        self,
        voice_name,
    ):

        if voice_name not in self.voices:

            raise ValueError(
                f"Voice not found: "
                f"{voice_name}"
            )

        self.current_voice = (
            voice_name
        )

    def speak(self, text):

        if not text:
            return

        text = str(text).strip()

        if not text:
            return

        try:

            while True:

                self.speech_queue.get_nowait()

        except queue.Empty:

            pass

        try:

            self.speech_queue.put_nowait(
                text
            )

        except queue.Full:

            pass

    def _worker_loop(self):

        while not self.stop_event.is_set():

            try:

                text = (
                    self.speech_queue.get(
                        timeout=0.1
                    )
                )

            except queue.Empty:

                continue

            try:

                self._speak_blocking(
                    text
                )

            except Exception as error:

                print(
                    "TTS error:",
                    error,
                )

    def _speak_blocking(
        self,
        text,
    ):

        voice_name = (
            self.current_voice
        )

        if voice_name is None:
            return

        voice = self.voices[
            voice_name
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

            if self.stop_event.is_set():
                return

            self.current_output = (
                output_path
            )

            winsound.PlaySound(
                str(output_path),
                winsound.SND_FILENAME,
            )

        finally:

            winsound.PlaySound(
                None,
                winsound.SND_PURGE,
            )

            if output_path.exists():

                try:

                    output_path.unlink()

                except PermissionError:

                    pass

            self.current_output = None

    def stop(self):

        self.stop_event.set()

        winsound.PlaySound(
            None,
            winsound.SND_PURGE,
        )

        try:

            while True:

                self.speech_queue.get_nowait()

        except queue.Empty:

            pass

        if (
            self.current_output
            is not None
            and self.current_output.exists()
        ):

            try:

                self.current_output.unlink()

            except PermissionError:

                pass

    def close(self):

        self.stop()

        if self.worker.is_alive():

            self.worker.join(
                timeout=1.0
            )