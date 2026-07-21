"""CARLA camera adapter with no import-time dependency on the CARLA package.

Create and attach the RGB sensor in CARLA code, then pass that sensor actor to
``CarlaSensorSource``.  The callback only converts and queues the newest frame;
perception and planning stay on the demo/main thread.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import replace
from typing import Callable, Optional

import numpy as np

from .external_source import FramePacket


def carla_image_to_bgr(image: object) -> np.ndarray:
    """Copy a CARLA BGRA image into an OpenCV BGR array."""
    width = int(getattr(image, "width"))
    height = int(getattr(image, "height"))
    if width <= 0 or height <= 0:
        raise ValueError("CARLA image dimensions must be positive")
    raw = np.frombuffer(getattr(image, "raw_data"), dtype=np.uint8)
    expected = width * height * 4
    if raw.size != expected:
        raise ValueError(f"CARLA BGRA buffer has {raw.size} bytes, expected {expected}")
    bgra = raw.reshape((height, width, 4))
    return bgra[:, :, :3].copy()


class CarlaSensorSource:
    """Latest-only queue around an already-created CARLA RGB sensor actor."""

    def __init__(
        self,
        sensor: object,
        *,
        timeout_sec: float = 0.5,
        episode_id: str = "default",
        owns_sensor: bool = False,
        wall_clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        parsed_timeout = float(timeout_sec)
        if not math.isfinite(parsed_timeout) or parsed_timeout <= 0.0:
            raise ValueError("CARLA source timeout_sec must be positive and finite")
        if not callable(getattr(sensor, "listen", None)):
            raise TypeError("CARLA sensor actor must provide listen(callback)")
        self._sensor = sensor
        self._timeout_sec = parsed_timeout
        self._episode_id = str(episode_id)
        self._owns_sensor = bool(owns_sensor)
        self._wall_clock = wall_clock
        self._monotonic_clock = monotonic_clock
        self._queue: queue.Queue[tuple[int, float, FramePacket]] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._closed = False
        self._reset_pending = True
        self._episode_generation = 0
        self._last_enqueued_frame = -1
        self._last_enqueued_timestamp = -1.0
        self._last_valid_monotonic: Optional[float] = None
        self._loss_started_monotonic: Optional[float] = None
        self._last_source_timestamp = 0.0
        self._sensor.listen(self._on_image)

    @property
    def description(self) -> str:
        return f"carla:sensor:{getattr(self._sensor, 'id', 'unknown')}"

    @property
    def nominal_fps(self) -> float:
        return 0.0

    def _on_image(self, image: object) -> None:
        try:
            arrival_monotonic = float(self._monotonic_clock())
            if not math.isfinite(arrival_monotonic):
                return
            with self._lock:
                if self._closed:
                    return
                generation = self._episode_generation
            frame_id = int(getattr(image, "frame"))
            timestamp = float(getattr(image, "timestamp"))
            if frame_id < 0 or not math.isfinite(timestamp) or timestamp < 0.0:
                return
            frame = carla_image_to_bgr(image)
            with self._lock:
                if self._closed or generation != self._episode_generation:
                    return
                if (
                    frame_id <= self._last_enqueued_frame
                    or timestamp <= self._last_enqueued_timestamp
                ):
                    return
                self._last_enqueued_frame = frame_id
                self._last_enqueued_timestamp = timestamp
                episode_id = self._episode_id
                packet = FramePacket(
                    frame=frame,
                    wall_timestamp=float(self._wall_clock()),
                    sequence_timestamp=timestamp,
                    frame_id=frame_id,
                    source_kind="carla",
                    valid=True,
                    episode_id=episode_id,
                )
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self._queue.put_nowait((generation, arrival_monotonic, packet))
        except (AttributeError, TypeError, ValueError, queue.Full):
            return

    def _invalid_packet_locked(
        self,
        now_monotonic: float,
        *,
        minimum_age_sec: float = 0.0,
    ) -> FramePacket:
        reset = self._reset_pending
        self._reset_pending = False
        minimum_age = max(0.0, float(minimum_age_sec))
        if self._last_valid_monotonic is not None:
            baseline = self._last_valid_monotonic
        else:
            if self._loss_started_monotonic is None:
                self._loss_started_monotonic = now_monotonic - minimum_age
            baseline = self._loss_started_monotonic
        age = max(minimum_age, now_monotonic - baseline)
        return FramePacket(
            frame=None,
            wall_timestamp=float(self._wall_clock()),
            sequence_timestamp=self._last_source_timestamp,
            frame_id=self._last_enqueued_frame,
            source_kind="carla",
            valid=False,
            reset=reset,
            perception_age_sec=age,
            episode_id=self._episode_id,
        )

    def read(self) -> FramePacket:
        deadline = time.monotonic() + self._timeout_sec
        while True:
            with self._lock:
                if self._closed:
                    raise RuntimeError("CARLA source is closed")
            remaining = max(0.0, deadline - time.monotonic())
            try:
                generation, arrival_monotonic, packet = self._queue.get(
                    timeout=remaining
                )
            except queue.Empty:
                now_monotonic = float(self._monotonic_clock())
                with self._lock:
                    if self._closed:
                        raise RuntimeError("CARLA source is closed")
                    return self._invalid_packet_locked(now_monotonic)
            with self._lock:
                if generation != self._episode_generation:
                    continue
                now_monotonic = float(self._monotonic_clock())
                age = max(0.0, now_monotonic - arrival_monotonic)
                if age > self._timeout_sec:
                    return self._invalid_packet_locked(
                        now_monotonic,
                        minimum_age_sec=age,
                    )
                deliver_reset = self._reset_pending
                self._reset_pending = False
                self._last_valid_monotonic = arrival_monotonic
                self._loss_started_monotonic = None
                self._last_source_timestamp = packet.sequence_timestamp
                return replace(
                    packet,
                    reset=deliver_reset,
                    perception_age_sec=age,
                )

    def begin_episode(self, episode_id: str) -> None:
        with self._lock:
            self._episode_id = str(episode_id)
            self._episode_generation += 1
            self._reset_pending = True
            self._last_enqueued_frame = -1
            self._last_enqueued_timestamp = -1.0
            self._last_valid_monotonic = None
            self._loss_started_monotonic = None
            self._last_source_timestamp = 0.0
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

    def reset(self) -> bool:
        with self._lock:
            if self._closed:
                return False
            episode_id = self._episode_id
        self.begin_episode(episode_id)
        return True

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        stop = getattr(self._sensor, "stop", None)
        try:
            if callable(stop):
                stop()
        finally:
            if self._owns_sensor:
                destroy = getattr(self._sensor, "destroy", None)
                if callable(destroy):
                    destroy()


__all__ = ["CarlaSensorSource", "carla_image_to_bgr"]
