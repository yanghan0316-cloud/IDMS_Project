"""Timestamp-safe front-camera sources for RSSM inference.

Host wall time is kept separate from the monotonic/media/simulator sequence
time consumed by the recurrent world model.  Implementations must never turn
a source timeout into a valid empty-road observation.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol, runtime_checkable

import cv2
import numpy as np


@dataclass(frozen=True)
class FramePacket:
    """One front-camera observation or an explicit source-state event."""

    frame: Optional[np.ndarray]
    wall_timestamp: float
    sequence_timestamp: float
    frame_id: int
    source_kind: str
    valid: bool = True
    eof: bool = False
    reset: bool = False
    perception_age_sec: float = 0.0
    episode_id: str = ""

    def __post_init__(self) -> None:
        numeric = (
            self.wall_timestamp,
            self.sequence_timestamp,
            self.perception_age_sec,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("frame packet timestamps and age must be finite")
        if self.sequence_timestamp < 0.0 or self.perception_age_sec < 0.0:
            raise ValueError("frame packet sequence time and age must be non-negative")
        if isinstance(self.frame_id, bool) or int(self.frame_id) != self.frame_id:
            raise ValueError("frame_id must be an integer")
        if self.eof and self.valid:
            raise ValueError("EOF cannot be a valid observation")
        if self.valid:
            if not isinstance(self.frame, np.ndarray):
                raise ValueError("a valid frame packet requires a numpy frame")
            if self.frame.dtype != np.uint8 or self.frame.ndim != 3 or self.frame.shape[2] != 3:
                raise ValueError("frame must be uint8 HxWx3 BGR")
        elif self.frame is not None:
            raise ValueError("an invalid frame packet must not carry a frame")


@runtime_checkable
class ExternalFrameSource(Protocol):
    """Minimal source boundary also implemented by the CARLA queue adapter."""

    @property
    def description(self) -> str: ...

    @property
    def nominal_fps(self) -> float: ...

    def read(self) -> FramePacket: ...

    def reset(self) -> bool: ...

    def close(self) -> None: ...


def parse_camera_source(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("camera source must be an integer index")
    if isinstance(value, int):
        index = value
    else:
        text = str(value).strip()
        if not text or not text.lstrip("+").isdigit():
            raise ValueError("camera source must be an integer index")
        index = int(text)
    if index < 0:
        raise ValueError("camera source must be non-negative")
    return index


class OpenCVFrameSource:
    """Open a live camera or a video while preserving the source clock."""

    def __init__(
        self,
        mode: str,
        source: object,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        capture_factory: Callable[[object], object] = cv2.VideoCapture,
        wall_clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
        latest_camera: bool = True,
        camera_timeout_sec: float = 1.0,
    ) -> None:
        self.mode = str(mode).strip().lower()
        if self.mode not in {"camera", "video"}:
            raise ValueError("OpenCV source mode must be 'camera' or 'video'")
        if not isinstance(latest_camera, bool):
            raise TypeError("latest_camera must be a bool")
        parsed_camera_timeout = float(camera_timeout_sec)
        if not math.isfinite(parsed_camera_timeout) or parsed_camera_timeout <= 0.0:
            raise ValueError("camera_timeout_sec must be positive and finite")
        if self.mode == "camera":
            self._source = parse_camera_source(source)
        else:
            path = Path(str(source)).expanduser()
            if capture_factory is cv2.VideoCapture and not path.is_file():
                raise FileNotFoundError(f"front-camera video not found: {path}")
            self._source = str(path)

        self._wall_clock = wall_clock
        self._monotonic_clock = monotonic_clock
        self._capture = capture_factory(self._source)
        if self._capture is None or not self._capture.isOpened():
            raise RuntimeError(f"cannot open {self.mode} source: {self._source}")
        if self.mode == "camera":
            # Backends may ignore this property, so the default capture thread
            # below also continuously drains the device into a latest-only slot.
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if width is not None and int(width) > 0:
                self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            if height is not None and int(height) > 0:
                self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

        raw_fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self._nominal_fps = raw_fps if math.isfinite(raw_fps) and raw_fps > 0.0 else 0.0
        self._frame_id = 0
        self._last_sequence_timestamp: Optional[float] = None
        self._last_valid_monotonic: Optional[float] = None
        self._loss_started_monotonic: Optional[float] = None
        self._reset_next = False
        self._closed = False
        self._latest_camera = self.mode == "camera" and latest_camera
        self._camera_timeout_sec = parsed_camera_timeout
        self._camera_condition = threading.Condition()
        self._camera_latest: Optional[
            tuple[int, bool, Optional[np.ndarray], float, float]
        ] = None
        self._camera_capture_serial = 0
        self._camera_delivered_serial = 0
        self._camera_thread: Optional[threading.Thread] = None
        if self._latest_camera:
            self._camera_thread = threading.Thread(
                target=self._camera_capture_loop,
                name=f"OpenCVFrameSource-{self._source}",
                daemon=True,
            )
            self._camera_thread.start()

    @property
    def description(self) -> str:
        return f"opencv:{self.mode}:{self._source}"

    @property
    def nominal_fps(self) -> float:
        return self._nominal_fps

    def _next_sequence_timestamp(self, capture_monotonic: float) -> float:
        if self.mode == "camera":
            candidate = capture_monotonic
        else:
            pts = float(self._capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
            candidate = pts if math.isfinite(pts) and pts >= 0.0 else 0.0
            if self._last_sequence_timestamp is not None and candidate <= self._last_sequence_timestamp:
                step = 1.0 / self._nominal_fps if self._nominal_fps > 0.0 else 1.0 / 25.0
                candidate = self._last_sequence_timestamp + step
        if self._last_sequence_timestamp is not None and candidate <= self._last_sequence_timestamp:
            candidate = math.nextafter(self._last_sequence_timestamp, math.inf)
        return candidate

    def _camera_capture_loop(self) -> None:
        while True:
            with self._camera_condition:
                if self._closed:
                    return
            try:
                ok, frame = self._capture.read()
            except Exception:
                ok, frame = False, None
            captured_monotonic = float(self._monotonic_clock())
            wall_timestamp = float(self._wall_clock())
            with self._camera_condition:
                if self._closed:
                    return
                self._camera_capture_serial += 1
                self._camera_latest = (
                    self._camera_capture_serial,
                    bool(ok),
                    frame,
                    captured_monotonic,
                    wall_timestamp,
                )
                self._camera_condition.notify_all()
            if not ok:
                time.sleep(0.01)

    def _read_capture(
        self,
    ) -> tuple[bool, Optional[np.ndarray], float, float, float]:
        if not self._latest_camera:
            ok, frame = self._capture.read()
            captured_monotonic = float(self._monotonic_clock())
            wall_timestamp = float(self._wall_clock())
            return bool(ok), frame, captured_monotonic, wall_timestamp, 0.0

        deadline = time.monotonic() + self._camera_timeout_sec
        with self._camera_condition:
            while (
                self._camera_latest is None
                or self._camera_latest[0] <= self._camera_delivered_serial
            ):
                if self._closed:
                    raise RuntimeError("frame source is closed")
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    captured_monotonic = float(self._monotonic_clock())
                    wall_timestamp = float(self._wall_clock())
                    return False, None, captured_monotonic, wall_timestamp, 0.0
                self._camera_condition.wait(timeout=remaining)
            serial, ok, frame, captured_monotonic, wall_timestamp = self._camera_latest
            self._camera_delivered_serial = serial
        delivered_monotonic = float(self._monotonic_clock())
        queue_age = max(0.0, delivered_monotonic - captured_monotonic)
        return ok, frame, captured_monotonic, wall_timestamp, queue_age

    def read(self) -> FramePacket:
        if self._closed:
            raise RuntimeError("frame source is closed")
        ok, frame, captured_monotonic, wall_timestamp, queue_age = self._read_capture()
        if self.mode == "camera" and queue_age > self._camera_timeout_sec:
            ok, frame = False, None
        if not ok or frame is None:
            if self.mode == "video":
                return FramePacket(
                    frame=None,
                    wall_timestamp=wall_timestamp,
                    sequence_timestamp=max(0.0, self._last_sequence_timestamp or 0.0),
                    frame_id=self._frame_id,
                    source_kind=self.mode,
                    valid=False,
                    eof=True,
                )
            observation_time = captured_monotonic + queue_age
            if self._loss_started_monotonic is None:
                baseline = self._last_valid_monotonic
                if baseline is None:
                    baseline = observation_time - queue_age
                self._loss_started_monotonic = baseline
            age = max(
                queue_age,
                observation_time - self._loss_started_monotonic,
            )
            sequence_timestamp = max(
                0.0,
                captured_monotonic,
                self._last_sequence_timestamp or 0.0,
            )
            return FramePacket(
                frame=None,
                wall_timestamp=wall_timestamp,
                sequence_timestamp=sequence_timestamp,
                frame_id=self._frame_id,
                source_kind=self.mode,
                valid=False,
                perception_age_sec=age,
            )

        if not isinstance(frame, np.ndarray):
            raise ValueError("OpenCV returned a non-array frame")
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("OpenCV frame must be uint8 HxWx3 BGR")
        sequence_timestamp = self._next_sequence_timestamp(captured_monotonic)
        packet = FramePacket(
            frame=frame,
            wall_timestamp=wall_timestamp,
            sequence_timestamp=sequence_timestamp,
            frame_id=self._frame_id,
            source_kind=self.mode,
            valid=True,
            reset=self._reset_next,
            perception_age_sec=queue_age,
        )
        self._frame_id += 1
        self._last_sequence_timestamp = sequence_timestamp
        self._last_valid_monotonic = captured_monotonic
        self._loss_started_monotonic = None
        self._reset_next = False
        return packet

    def reset(self) -> bool:
        if self._closed or self.mode != "video":
            return False
        if not self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0):
            return False
        self._frame_id = 0
        self._last_sequence_timestamp = None
        self._last_valid_monotonic = None
        self._loss_started_monotonic = None
        self._reset_next = True
        return True

    def close(self) -> None:
        with self._camera_condition:
            if self._closed:
                return
            self._closed = True
            self._camera_condition.notify_all()
        release_error: Optional[Exception] = None
        try:
            self._capture.release()
        except Exception as exc:
            release_error = exc
        thread = self._camera_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.1, self._camera_timeout_sec))
            if thread.is_alive():
                raise RuntimeError(
                    "camera capture thread did not stop after release"
                ) from release_error
        if release_error is not None:
            raise release_error


__all__ = [
    "ExternalFrameSource",
    "FramePacket",
    "OpenCVFrameSource",
    "parse_camera_source",
]
