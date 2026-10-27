"""
Functionality:
-------------
This module benchmarks the current multimodal ISL recognition
pipeline without changing the trained model or recognition logic.

The benchmark:
1. Loads the current multimodal GRU model.
2. Loads the dynamic class mapping.
3. Opens the webcam.
4. Runs the existing multimodal tracker.
5. Runs the existing unified feature extractor.
6. Maintains a 30-frame sequence buffer.
7. Runs GRU inference every 3 frames.
8. Measures webcam, tracking, feature extraction, inference,
   total processing, FPS, CPU, RAM, and optional GPU usage.
9. Saves frame-level measurements.
10. Saves a summary benchmark report.
"""

from collections import deque
from pathlib import Path
import csv
import json
import subprocess
import time
import sys

import cv2
import numpy as np
import tensorflow as tf

try:
    import psutil
except ImportError:
    psutil = None

from src.multimodal_tracker import MultimodalTracker
from src.feature_extractor import FeatureExtractor

SEQUENCE_LENGTH = 30
FEATURE_COUNT = 1662
INFERENCE_INTERVAL = 3
BENCHMARK_FRAMES = 300

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "gru_multimodal.keras"
)

DATASET_ROOT = Path(r"D:\ISL-Dataset")

PROCESSED_ROOT = (
    DATASET_ROOT
    / "extracted"
    / "processed_multimodal"
)

CLASSES_PATH = (
    PROCESSED_ROOT
    / "classes.json"
)

RESULTS_ROOT = (
    PROCESSED_ROOT
    / "results"
    / "baseline_benchmark"
)

FRAME_RESULTS_PATH = (
    RESULTS_ROOT
    / "frame_measurements.csv"
)

SUMMARY_PATH = (
    RESULTS_ROOT
    / "baseline_benchmark_summary.json"
)


def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


def load_classes():

    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Class mapping not found at: {CLASSES_PATH}"
        )

    with open(
        CLASSES_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        classes = json.load(file)

    if not isinstance(classes, list):
        raise ValueError(
            "classes.json must contain a list."
        )

    if not classes:
        raise ValueError(
            "classes.json contains no classes."
        )

    return classes


def validate_model(
    model,
    classes,
):

    output_classes = model.output_shape[-1]

    if output_classes != len(classes):
        raise ValueError(
            "Model output classes do not match "
            f"class mapping. Model: {output_classes}, "
            f"Classes: {len(classes)}"
        )

    input_shape = model.input_shape

    expected_shape = (
        None,
        SEQUENCE_LENGTH,
        FEATURE_COUNT,
    )

    if tuple(input_shape) != expected_shape:
        raise ValueError(
            "Unexpected model input shape. "
            f"Model: {input_shape}, "
            f"Expected: {expected_shape}"
        )


def warm_up_model(model):

    warmup_input = np.zeros(
        (
            1,
            SEQUENCE_LENGTH,
            FEATURE_COUNT,
        ),
        dtype=np.float32,
    )

    for _ in range(5):

        model.predict(
            warmup_input,
            verbose=0,
        )


def detect_nvidia_smi():

    try:

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )

        output = result.stdout.strip()

        if not output:
            return None

        first_gpu = output.splitlines()[0]

        gpu_usage, gpu_memory = (
            value.strip()
            for value in first_gpu.split(",")
        )

        return {
            "gpu_utilization_percent": float(
                gpu_usage
            ),
            "gpu_memory_used_mb": float(
                gpu_memory
            ),
        }

    except (
        FileNotFoundError,
        subprocess.SubprocessError,
        ValueError,
    ):
        return None


def get_process_metrics():

    if psutil is None:
        return {
            "cpu_percent": None,
            "ram_mb": None,
        }

    process = psutil.Process()

    return {
        "cpu_percent": process.cpu_percent(
            interval=None
        ),
        "ram_mb": (
            process.memory_info().rss
            / (1024 ** 2)
        ),
    }


def mean_or_none(values):

    if not values:
        return None

    return float(
        np.mean(values)
    )


def min_or_none(values):

    if not values:
        return None

    return float(
        np.min(values)
    )


def max_or_none(values):

    if not values:
        return None

    return float(
        np.max(values)
    )


def main():

    print()
    print(
        "Starting baseline benchmark..."
    )
    print()

    if psutil is None:
        print(
            "WARNING: psutil is not installed."
        )
        print(
            "CPU and RAM measurements will be unavailable."
        )
        print()

    model = load_model()
    classes = load_classes()

    validate_model(
        model,
        classes,
    )

    print(
        f"Model classes: {len(classes)}"
    )

    print(
        f"Model input: {SEQUENCE_LENGTH} x "
        f"{FEATURE_COUNT}"
    )

    print(
        "Warming up model..."
    )

    warm_up_model(
        model
    )

    print(
        "Model warm-up completed."
    )

    tracker = MultimodalTracker()
    extractor = FeatureExtractor()

    sequence_buffer = deque(
        maxlen=SEQUENCE_LENGTH
    )

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not camera.isOpened():

        tracker.close()

        raise RuntimeError(
            "Could not open webcam."
        )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT,
    )

    actual_width = int(
        camera.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    actual_height = int(
        camera.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print(
        f"Camera resolution: "
        f"{actual_width} x {actual_height}"
    )

    print(
        f"Target benchmark frames: "
        f"{BENCHMARK_FRAMES}"
    )

    print()
    print(
        "Press Q to stop early."
    )
    print()

    frame_records = []

    tracking_times = []
    feature_times = []
    inference_times = []
    frame_processing_times = []
    capture_times = []

    cpu_values = []
    ram_values = []
    gpu_values = []
    gpu_memory_values = []

    inference_count = 0
    successful_frames = 0

    frame_counter = 0
    last_inference_frame = 0

    benchmark_start = time.perf_counter()

    process_metric_state = get_process_metrics()

    try:

        while (
            frame_counter
            < BENCHMARK_FRAMES
        ):

            frame_start = (
                time.perf_counter()
            )

            capture_start = (
                time.perf_counter()
            )

            success, frame = camera.read()

            capture_end = (
                time.perf_counter()
            )

            capture_ms = (
                capture_end
                - capture_start
            ) * 1000.0

            if not success:

                print(
                    "Failed to read frame."
                )

                break

            frame = cv2.flip(
                frame,
                1,
            )

            tracking_start = (
                time.perf_counter()
            )

            timestamp_ms = int(
                (
                    time.perf_counter()
                    - benchmark_start
                )
                * 1000
            )

            tracking_result = tracker.process(
                frame,
                timestamp_ms,
            )

            tracking_end = (
                time.perf_counter()
            )

            tracking_ms = (
                tracking_end
                - tracking_start
            ) * 1000.0

            feature_start = (
                time.perf_counter()
            )

            features = extractor.extract(
                tracking_result
            )

            features = np.asarray(
                features,
                dtype=np.float32,
            )

            feature_end = (
                time.perf_counter()
            )

            feature_ms = (
                feature_end
                - feature_start
            ) * 1000.0

            if features.shape != (
                FEATURE_COUNT,
            ):

                raise ValueError(
                    "Unexpected feature shape: "
                    f"{features.shape}. "
                    f"Expected: "
                    f"({FEATURE_COUNT},)"
                )

            sequence_buffer.append(
                features
            )

            inference_ms = None
            predicted_class = None
            confidence = None

            frame_counter += 1

            if (
                len(sequence_buffer)
                == SEQUENCE_LENGTH
                and (
                    frame_counter
                    - last_inference_frame
                    >= INFERENCE_INTERVAL
                )
            ):

                sequence = np.asarray(
                    sequence_buffer,
                    dtype=np.float32,
                )

                sequence = np.expand_dims(
                    sequence,
                    axis=0,
                )

                inference_start = (
                    time.perf_counter()
                )

                probabilities = (
                    model.predict(
                        sequence,
                        verbose=0,
                    )[0]
                )

                inference_end = (
                    time.perf_counter()
                )

                inference_ms = (
                    inference_end
                    - inference_start
                ) * 1000.0

                predicted_index = int(
                    np.argmax(
                        probabilities
                    )
                )

                confidence = float(
                    probabilities[
                        predicted_index
                    ]
                )

                predicted_class = classes[
                    predicted_index
                ]

                inference_times.append(
                    inference_ms
                )

                inference_count += 1

                last_inference_frame = (
                    frame_counter
                )

                gpu_metrics = (
                    detect_nvidia_smi()
                )

                if gpu_metrics is not None:

                    gpu_values.append(
                        gpu_metrics[
                            "gpu_utilization_percent"
                        ]
                    )

                    gpu_memory_values.append(
                        gpu_metrics[
                            "gpu_memory_used_mb"
                        ]
                    )

            frame_end = (
                time.perf_counter()
            )

            frame_processing_ms = (
                frame_end
                - frame_start
            ) * 1000.0

            process_metric_state = (
                get_process_metrics()
            )

            if (
                process_metric_state[
                    "cpu_percent"
                ]
                is not None
            ):

                cpu_values.append(
                    process_metric_state[
                        "cpu_percent"
                    ]
                )

                ram_values.append(
                    process_metric_state[
                        "ram_mb"
                    ]
                )

            capture_times.append(
                capture_ms
            )

            tracking_times.append(
                tracking_ms
            )

            feature_times.append(
                feature_ms
            )

            frame_processing_times.append(
                frame_processing_ms
            )

            successful_frames += 1

            frame_records.append(
                {
                    "frame": frame_counter,
                    "capture_ms": capture_ms,
                    "tracking_ms": tracking_ms,
                    "feature_ms": feature_ms,
                    "inference_ms": inference_ms,
                    "frame_processing_ms": (
                        frame_processing_ms
                    ),
                    "buffer_length": len(
                        sequence_buffer
                    ),
                    "predicted_class": (
                        predicted_class
                    ),
                    "confidence": confidence,
                    "cpu_percent": (
                        process_metric_state[
                            "cpu_percent"
                        ]
                    ),
                    "ram_mb": (
                        process_metric_state[
                            "ram_mb"
                        ]
                    ),
                }
            )

            cv2.putText(
                frame,
                "BASELINE BENCHMARK",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Frame: {frame_counter}/"
                f"{BENCHMARK_FRAMES}",
                (30, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Buffer: "
                f"{len(sequence_buffer)}/"
                f"{SEQUENCE_LENGTH}",
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if predicted_class is not None:

                cv2.putText(
                    frame,
                    f"Prediction: "
                    f"{predicted_class}",
                    (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    f"Confidence: "
                    f"{confidence * 100:.1f}%",
                    (30, 195),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                "Q: Quit",
                (30, 235),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Baseline Benchmark",
                frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

    finally:

        camera.release()

        cv2.destroyAllWindows()

        tracker.close()

    benchmark_end = (
        time.perf_counter()
    )

    total_duration = (
        benchmark_end
        - benchmark_start
    )

    average_fps = (
        successful_frames
        / total_duration
        if total_duration > 0
        else 0.0
    )

    summary = {
        "benchmark": {
            "frames_requested": (
                BENCHMARK_FRAMES
            ),
            "frames_processed": (
                successful_frames
            ),
            "benchmark_duration_seconds": (
                total_duration
            ),
        },
        "model": {
            "model_path": str(
                MODEL_PATH
            ),
            "input_shape": [
                SEQUENCE_LENGTH,
                FEATURE_COUNT,
            ],
            "class_count": len(classes),
            "inference_interval_frames": (
                INFERENCE_INTERVAL
            ),
            "inference_count": (
                inference_count
            ),
        },
        "camera": {
            "requested_width": (
                CAMERA_WIDTH
            ),
            "requested_height": (
                CAMERA_HEIGHT
            ),
            "actual_width": actual_width,
            "actual_height": actual_height,
        },
        "performance": {
            "average_fps": average_fps,
            "average_capture_ms": (
                mean_or_none(
                    capture_times
                )
            ),
            "average_tracking_ms": (
                mean_or_none(
                    tracking_times
                )
            ),
            "average_feature_ms": (
                mean_or_none(
                    feature_times
                )
            ),
            "average_inference_ms": (
                mean_or_none(
                    inference_times
                )
            ),
            "minimum_inference_ms": (
                min_or_none(
                    inference_times
                )
            ),
            "maximum_inference_ms": (
                max_or_none(
                    inference_times
                )
            ),
            "average_frame_processing_ms": (
                mean_or_none(
                    frame_processing_times
                )
            ),
            "minimum_frame_processing_ms": (
                min_or_none(
                    frame_processing_times
                )
            ),
            "maximum_frame_processing_ms": (
                max_or_none(
                    frame_processing_times
                )
            ),
        },
        "resources": {
            "average_process_cpu_percent": (
                mean_or_none(
                    cpu_values
                )
            ),
            "maximum_process_cpu_percent": (
                max_or_none(
                    cpu_values
                )
            ),
            "average_process_ram_mb": (
                mean_or_none(
                    ram_values
                )
            ),
            "maximum_process_ram_mb": (
                max_or_none(
                    ram_values
                )
            ),
            "average_gpu_utilization_percent": (
                mean_or_none(
                    gpu_values
                )
            ),
            "maximum_gpu_utilization_percent": (
                max_or_none(
                    gpu_values
                )
            ),
            "average_gpu_memory_used_mb": (
                mean_or_none(
                    gpu_memory_values
                )
            ),
            "maximum_gpu_memory_used_mb": (
                max_or_none(
                    gpu_memory_values
                )
            ),
            "psutil_available": (
                psutil is not None
            ),
            "nvidia_smi_available": (
                bool(gpu_values)
            ),
        },
    }

    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        FRAME_RESULTS_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fieldnames = [
            "frame",
            "capture_ms",
            "tracking_ms",
            "feature_ms",
            "inference_ms",
            "frame_processing_ms",
            "buffer_length",
            "predicted_class",
            "confidence",
            "cpu_percent",
            "ram_mb",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            frame_records
        )

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print()
    print(
        "========== BASELINE BENCHMARK =========="
    )
    print(
        f"Frames processed: "
        f"{successful_frames}"
    )
    print(
        f"Total duration: "
        f"{total_duration:.2f} s"
    )
    print(
        f"Average FPS: "
        f"{average_fps:.2f}"
    )
    print(
        f"Average tracking: "
        f"{mean_or_none(tracking_times):.2f} ms"
    )
    print(
        f"Average features: "
        f"{mean_or_none(feature_times):.2f} ms"
    )
    print(
        f"Average inference: "
        f"{mean_or_none(inference_times):.2f} ms"
    )
    print(
        f"Average frame processing: "
        f"{mean_or_none(frame_processing_times):.2f} ms"
    )

    if cpu_values:

        print(
            f"Average process CPU: "
            f"{mean_or_none(cpu_values):.2f}%"
        )

        print(
            f"Maximum process CPU: "
            f"{max_or_none(cpu_values):.2f}%"
        )

        print(
            f"Average process RAM: "
            f"{mean_or_none(ram_values):.2f} MB"
        )

        print(
            f"Maximum process RAM: "
            f"{max_or_none(ram_values):.2f} MB"
        )

    else:

        print(
            "CPU/RAM: unavailable"
        )

    if gpu_values:

        print(
            f"Average GPU utilization: "
            f"{mean_or_none(gpu_values):.2f}%"
        )

        print(
            f"Maximum GPU utilization: "
            f"{max_or_none(gpu_values):.2f}%"
        )

        print(
            f"Average GPU memory: "
            f"{mean_or_none(gpu_memory_values):.2f} MB"
        )

        print(
            f"Maximum GPU memory: "
            f"{max_or_none(gpu_memory_values):.2f} MB"
        )

    else:

        print(
            "GPU metrics: unavailable"
        )

    print()
    print(
        f"Frame measurements saved to:"
    )
    print(
        FRAME_RESULTS_PATH
    )

    print()
    print(
        "Summary saved to:"
    )
    print(
        SUMMARY_PATH
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()