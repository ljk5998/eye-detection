"""Small prototype for iris-center tracking and CSV logging.

This script is intentionally narrow: it tracks left/right iris centers with
MediaPipe Face Mesh, estimates frame-to-frame velocity, overlays the result,
and saves raw screening metrics for later threshold design.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

CSV_FIELDS = [
    "frame_index",
    "timestamp_sec",
    "face_detected",
    "left_iris_x",
    "left_iris_y",
    "left_rel_x",
    "left_rel_y",
    "left_dx",
    "left_dy",
    "left_velocity",
    "right_iris_x",
    "right_iris_y",
    "right_rel_x",
    "right_rel_y",
    "right_dx",
    "right_dy",
    "right_velocity",
    "mean_velocity",
]


@dataclass(frozen=True)
class EyeSpec:
    name: str
    iris_indices: tuple[int, int, int, int]
    outer_corner: int
    inner_corner: int
    upper_lid: int
    lower_lid: int
    color_bgr: tuple[int, int, int]


@dataclass
class MotionState:
    previous_center: tuple[float, float] | None = None
    previous_time: float | None = None


EYE_SPECS = (
    EyeSpec(
        name="right",
        iris_indices=(469, 470, 471, 472),
        outer_corner=33,
        inner_corner=133,
        upper_lid=159,
        lower_lid=145,
        color_bgr=(0, 200, 255),
    ),
    EyeSpec(
        name="left",
        iris_indices=(474, 475, 476, 477),
        outer_corner=263,
        inner_corner=362,
        upper_lid=386,
        lower_lid=374,
        color_bgr=(255, 180, 0),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track MediaPipe iris centers and save screening metrics."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index such as 0, or a video file path. Defaults to 0.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/iris_trace.csv",
        help="CSV output path for iris coordinates and velocity metrics.",
    )
    parser.add_argument(
        "--model",
        default="data/models/face_landmarker.task",
        help="Face Landmarker .task model path for newer MediaPipe Tasks API.",
    )
    parser.add_argument(
        "--no-auto-download-model",
        action="store_true",
        help="Do not download the Face Landmarker model when it is missing.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable OpenCV preview window and only write CSV.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for quick smoke tests.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=1280,
        help="Requested webcam capture width.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=720,
        help="Requested webcam capture height.",
    )
    return parser.parse_args()


def parse_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def load_dependencies() -> tuple[Any, Any]:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Run: pip install -r requirements.txt"
        ) from exc
    return cv2, mp


def configure_runtime_cache() -> None:
    cache_dir = Path("data/.cache/matplotlib")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir.resolve()))


def ensure_face_landmarker_model(model_path: Path, auto_download: bool) -> Path:
    if model_path.exists():
        return model_path

    if not auto_download:
        raise SystemExit(
            f"Missing model file: {model_path}\n"
            f"Download it from:\n{FACE_LANDMARKER_MODEL_URL}\n"
            f"or rerun without --no-auto-download-model."
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Face Landmarker model to {model_path} ...")
    try:
        urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, model_path)
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(
            f"Could not download model automatically: {exc}\n"
            f"Download manually:\n"
            f"curl -L {FACE_LANDMARKER_MODEL_URL} -o {model_path}"
        ) from exc

    return model_path


class LegacyFaceMeshDetector:
    def __init__(self, mp: Any) -> None:
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, rgb_frame: Any, timestamp_ms: int) -> list[Any] | None:
        del timestamp_ms
        rgb_frame.flags.writeable = False
        results = self.face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0].landmark

    def close(self) -> None:
        self.face_mesh.close()


class TasksFaceLandmarkerDetector:
    def __init__(self, mp: Any, model_path: Path) -> None:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(model_path),
                delegate=python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp = mp
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def process(self, rgb_frame: Any, timestamp_ms: int) -> list[Any] | None:
        mp_image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=rgb_frame,
        )
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]

    def close(self) -> None:
        self.landmarker.close()


def create_landmark_detector(
    mp: Any, model_path: Path, auto_download_model: bool
) -> LegacyFaceMeshDetector | TasksFaceLandmarkerDetector:
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        return LegacyFaceMeshDetector(mp)

    if not hasattr(mp, "tasks"):
        raise SystemExit(
            "This MediaPipe install exposes neither mp.solutions nor mp.tasks. "
            "Try reinstalling with: pip install --upgrade mediapipe"
        )

    model_path = ensure_face_landmarker_model(model_path, auto_download_model)
    return TasksFaceLandmarkerDetector(mp, model_path)


def landmark_to_pixel(landmark: Any, width: int, height: int) -> tuple[float, float]:
    return landmark.x * width, landmark.y * height


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def mean_point(
    landmarks: list[Any], indices: tuple[int, ...], width: int, height: int
) -> tuple[float, float]:
    points = [landmark_to_pixel(landmarks[index], width, height) for index in indices]
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def relative_position(
    landmarks: list[Any], spec: EyeSpec, center: tuple[float, float], width: int, height: int
) -> tuple[float, float]:
    outer = landmark_to_pixel(landmarks[spec.outer_corner], width, height)
    inner = landmark_to_pixel(landmarks[spec.inner_corner], width, height)
    upper = landmark_to_pixel(landmarks[spec.upper_lid], width, height)
    lower = landmark_to_pixel(landmarks[spec.lower_lid], width, height)

    eye_axis = (inner[0] - outer[0], inner[1] - outer[1])
    axis_length_sq = max(eye_axis[0] ** 2 + eye_axis[1] ** 2, 1.0)
    axis_length = math.sqrt(axis_length_sq)
    axis_unit = (eye_axis[0] / axis_length, eye_axis[1] / axis_length)
    vertical_unit = (-axis_unit[1], axis_unit[0])

    center_from_outer = (center[0] - outer[0], center[1] - outer[1])
    rel_x = (
        center_from_outer[0] * eye_axis[0] + center_from_outer[1] * eye_axis[1]
    ) / axis_length_sq

    lid_mid = ((upper[0] + lower[0]) / 2.0, (upper[1] + lower[1]) / 2.0)
    center_from_mid = (center[0] - lid_mid[0], center[1] - lid_mid[1])
    eye_height = max(distance(upper, lower), 1.0)
    rel_y = (
        center_from_mid[0] * vertical_unit[0] + center_from_mid[1] * vertical_unit[1]
    ) / eye_height
    return rel_x, rel_y


def compute_motion(
    state: MotionState, center: tuple[float, float], timestamp_sec: float
) -> dict[str, float | None]:
    if state.previous_center is None or state.previous_time is None:
        state.previous_center = center
        state.previous_time = timestamp_sec
        return {"dx": None, "dy": None, "velocity": None}

    dt = max(timestamp_sec - state.previous_time, 1e-6)
    dx = center[0] - state.previous_center[0]
    dy = center[1] - state.previous_center[1]
    velocity = math.hypot(dx, dy) / dt

    state.previous_center = center
    state.previous_time = timestamp_sec
    return {"dx": dx, "dy": dy, "velocity": velocity}


def empty_row(frame_index: int, timestamp_sec: float) -> dict[str, str | int]:
    row: dict[str, str | int] = {field: "" for field in CSV_FIELDS}
    row["frame_index"] = frame_index
    row["timestamp_sec"] = f"{timestamp_sec:.6f}"
    row["face_detected"] = 0
    return row


def format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def draw_eye_overlay(
    cv2: Any,
    frame: Any,
    spec: EyeSpec,
    center: tuple[float, float],
    rel: tuple[float, float],
    velocity: float | None,
    track: deque[tuple[int, int]],
) -> None:
    point = (int(round(center[0])), int(round(center[1])))
    track.append(point)

    cv2.circle(frame, point, 4, spec.color_bgr, -1)
    for start, end in zip(track, list(track)[1:]):
        cv2.line(frame, start, end, spec.color_bgr, 1)

    velocity_text = "--" if velocity is None else f"{velocity:.1f}px/s"
    label = f"{spec.name} iris ({rel[0]:.2f}, {rel[1]:.2f}) {velocity_text}"
    y = 28 if spec.name == "left" else 56
    cv2.putText(
        frame,
        label,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        spec.color_bgr,
        2,
        cv2.LINE_AA,
    )


def timestamp_for_frame(
    cv2: Any, capture: Any, frame_index: int, start_time: float, is_camera: bool
) -> float:
    if is_camera:
        return time.perf_counter() - start_time

    position_msec = capture.get(cv2.CAP_PROP_POS_MSEC)
    if position_msec > 0:
        return position_msec / 1000.0

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps > 0:
        return frame_index / fps

    return time.perf_counter() - start_time


def run() -> int:
    args = parse_args()
    configure_runtime_cache()
    cv2, mp = load_dependencies()

    source = parse_source(args.source)
    is_camera = isinstance(source, int)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"Could not open source: {args.source}", file=sys.stderr)
        return 1

    if is_camera:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    try:
        detector = create_landmark_detector(
            mp,
            Path(args.model),
            auto_download_model=not args.no_auto_download_model,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "kGpuService" in message or "NSOpenGLPixelFormat" in message:
            raise SystemExit(
                "MediaPipe Face Landmarker could not initialize the macOS "
                "graphics service. Run this command from a normal Terminal "
                "session with display access, or use a Python environment "
                "where the legacy mp.solutions.face_mesh API is available.\n"
                f"Original error: {exc}"
            ) from exc
        raise

    states = {spec.name: MotionState() for spec in EYE_SPECS}
    tracks = {spec.name: deque(maxlen=45) for spec in EYE_SPECS}
    max_landmark_index = max(
        max(spec.iris_indices + (spec.outer_corner, spec.inner_corner, spec.upper_lid, spec.lower_lid))
        for spec in EYE_SPECS
    )

    start_time = time.perf_counter()
    frame_index = 0

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                timestamp_sec = timestamp_for_frame(
                    cv2, capture, frame_index, start_time, is_camera
                )
                row = empty_row(frame_index, timestamp_sec)
                height, width = frame.shape[:2]

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = max(int(round(timestamp_sec * 1000)), frame_index)
                landmarks = detector.process(rgb_frame, timestamp_ms)
                row["face_detected"] = 1 if landmarks else 0

                if landmarks:
                    if len(landmarks) > max_landmark_index:
                        velocities: list[float] = []

                        for spec in EYE_SPECS:
                            center = mean_point(landmarks, spec.iris_indices, width, height)
                            rel = relative_position(landmarks, spec, center, width, height)
                            motion = compute_motion(states[spec.name], center, timestamp_sec)
                            velocity = motion["velocity"]
                            if velocity is not None:
                                velocities.append(float(velocity))

                            row[f"{spec.name}_iris_x"] = format_float(center[0])
                            row[f"{spec.name}_iris_y"] = format_float(center[1])
                            row[f"{spec.name}_rel_x"] = format_float(rel[0])
                            row[f"{spec.name}_rel_y"] = format_float(rel[1])
                            row[f"{spec.name}_dx"] = format_float(motion["dx"])
                            row[f"{spec.name}_dy"] = format_float(motion["dy"])
                            row[f"{spec.name}_velocity"] = format_float(velocity)

                            if not args.no_preview:
                                draw_eye_overlay(
                                    cv2,
                                    frame,
                                    spec,
                                    center,
                                    rel,
                                    velocity,
                                    tracks[spec.name],
                                )

                        if velocities:
                            row["mean_velocity"] = format_float(
                                sum(velocities) / len(velocities)
                            )
                    else:
                        row["face_detected"] = 0
                else:
                    for state in states.values():
                        state.previous_center = None
                        state.previous_time = None
                    for track in tracks.values():
                        track.clear()

                writer.writerow(row)

                if not args.no_preview:
                    cv2.putText(
                        frame,
                        "q/esc: quit",
                        (20, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (230, 230, 230),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("Iris Screening Prototype", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break

                frame_index += 1
                if args.max_frames is not None and frame_index >= args.max_frames:
                    break
        finally:
            detector.close()
            capture.release()
            if not args.no_preview:
                cv2.destroyAllWindows()

    print(f"Saved iris screening metrics to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
