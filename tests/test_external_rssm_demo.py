"""Hardware-free contracts for the external-only RSSM demo and adapters."""

from __future__ import annotations

import queue
import threading
import time
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from demo_rssm_external import (
    CABIN_STATUS,
    ExternalOnlyPipeline,
    ValidationStats,
    build_validation_outcome,
    parse_args,
    require_ready_planner,
    strict_validation_errors,
)
from src.core.risk_fusion import FusionResult
from src.external.collision_warn import CollisionWarner
from src.integration.carla_source import CarlaSensorSource, carla_image_to_bgr
from src.integration.external_source import FramePacket, OpenCVFrameSource


class _FakeCapture:
    def __init__(self, frames, *, fps=0.0, pts_ms=None):
        self.frames = list(frames)
        self.fps = fps
        self.pts_ms = list(pts_ms or [0.0] * len(self.frames))
        self.index = 0
        self.opened = True
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame.copy()

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.pts_ms[max(0, self.index - 1)]
        return 0.0

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES and int(value) == 0:
            self.index = 0
            return True
        return True

    def release(self):
        self.released = True


class _BlockingCapture:
    def __init__(self):
        self.items = queue.Queue()
        self.opened = True
        self.released = False
        self._reads = 0
        self._condition = threading.Condition()

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return 30.0 if prop == cv2.CAP_PROP_FPS else 0.0

    def set(self, prop, value):
        return True

    def push(self, frame):
        self.items.put((True, frame.copy()))

    def read(self):
        result = self.items.get(timeout=2.0)
        with self._condition:
            self._reads += 1
            self._condition.notify_all()
        return result

    def wait_for_reads(self, count):
        deadline = time.monotonic() + 1.0
        with self._condition:
            while self._reads < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(remaining)
        return True

    def release(self):
        self.released = True
        self.items.put((False, None))


class FrameSourceTests(unittest.TestCase):
    def test_frame_packet_rejects_invalid_shape_and_eof_valid_mix(self):
        with self.assertRaisesRegex(ValueError, "HxWx3"):
            FramePacket(np.zeros((2, 2), np.uint8), 1.0, 0.0, 0, "test")
        with self.assertRaisesRegex(ValueError, "EOF"):
            FramePacket(np.zeros((2, 2, 3), np.uint8), 1.0, 0.0, 0, "test", eof=True)

    def test_camera_uses_capture_monotonic_time_and_reports_loss(self):
        frame = np.zeros((3, 4, 3), np.uint8)
        capture = _FakeCapture([frame], fps=0.0)
        mono_values = iter((10.0, 10.4))
        wall_values = iter((100.0, 100.4))
        source = OpenCVFrameSource(
            "camera",
            "2",
            capture_factory=lambda _: capture,
            monotonic_clock=lambda: next(mono_values),
            latest_camera=False,
            wall_clock=lambda: next(wall_values),
        )

        valid = source.read()
        lost = source.read()

        self.assertTrue(valid.valid)
        self.assertEqual(valid.sequence_timestamp, 10.0)
        self.assertFalse(lost.valid)
        self.assertFalse(lost.eof)
        self.assertAlmostEqual(lost.perception_age_sec, 0.4)
        source.close()
        source.close()
        self.assertTrue(capture.released)

    def test_camera_loss_age_accumulates_before_any_valid_frame(self):
        capture = _FakeCapture([])
        mono_values = iter((10.0, 10.25))
        wall_values = iter((100.0, 100.25))
        source = OpenCVFrameSource(
            "camera",
            0,
            capture_factory=lambda _: capture,
            monotonic_clock=lambda: next(mono_values),
            wall_clock=lambda: next(wall_values),
            latest_camera=False,
        )

        first = source.read()
        second = source.read()

        self.assertEqual(first.perception_age_sec, 0.0)
        self.assertAlmostEqual(second.perception_age_sec, 0.25)
        source.close()

    def test_live_camera_capture_keeps_only_latest_frame(self):
        capture = _BlockingCapture()
        source = OpenCVFrameSource(
            "camera",
            0,
            capture_factory=lambda _: capture,
            camera_timeout_sec=0.2,
        )
        first = np.full((2, 2, 3), 1, np.uint8)
        latest = np.full((2, 2, 3), 2, np.uint8)
        capture.push(first)
        self.assertTrue(capture.wait_for_reads(1))
        capture.push(latest)
        self.assertTrue(capture.wait_for_reads(2))

        packet = source.read()

        self.assertTrue(packet.valid)
        self.assertEqual(packet.frame.tolist(), latest.tolist())
        self.assertGreaterEqual(packet.perception_age_sec, 0.0)
        source.close()
        self.assertTrue(capture.released)

    def test_camera_close_joins_thread_even_when_release_raises(self):
        class ReleaseFailureCapture(_BlockingCapture):
            def release(self):
                self.released = True
                self.items.put((False, None))
                raise RuntimeError("release failed")

        capture = ReleaseFailureCapture()
        source = OpenCVFrameSource(
            "camera",
            0,
            capture_factory=lambda _: capture,
            camera_timeout_sec=0.01,
        )

        with self.assertRaisesRegex(RuntimeError, "release failed"):
            source.close()

        self.assertFalse(source._camera_thread.is_alive())

    def test_camera_close_reports_a_capture_thread_that_will_not_stop(self):
        class StuckCapture(_BlockingCapture):
            def __init__(self):
                super().__init__()
                self.entered = threading.Event()

            def read(self):
                self.entered.set()
                return super().read()

            def release(self):
                self.released = True

            def unblock(self):
                self.items.put((False, None))

        capture = StuckCapture()
        source = OpenCVFrameSource(
            "camera",
            0,
            capture_factory=lambda _: capture,
            camera_timeout_sec=0.01,
        )
        self.assertTrue(capture.entered.wait(1.0))

        with self.assertRaisesRegex(RuntimeError, "thread did not stop"):
            source.close()

        self.assertTrue(source._camera_thread.is_alive())
        capture.unblock()
        source._camera_thread.join(1.0)
        self.assertFalse(source._camera_thread.is_alive())

    def test_live_camera_rejects_a_stale_latest_frame(self):
        capture = _BlockingCapture()
        monotonic_now = [10.0]
        source = OpenCVFrameSource(
            "camera",
            0,
            capture_factory=lambda _: capture,
            monotonic_clock=lambda: monotonic_now[0],
            wall_clock=lambda: 100.0,
            camera_timeout_sec=0.01,
        )
        capture.push(np.ones((2, 2, 3), np.uint8))
        self.assertTrue(capture.wait_for_reads(1))
        monotonic_now[0] = 10.02

        packet = source.read()

        self.assertFalse(packet.valid)
        self.assertIsNone(packet.frame)
        self.assertAlmostEqual(packet.perception_age_sec, 0.02)
        source.close()

    def test_video_duplicate_pts_gets_monotonic_fallback_and_eof(self):
        frame = np.zeros((2, 2, 3), np.uint8)
        capture = _FakeCapture([frame, frame], fps=20.0, pts_ms=[0.0, 0.0])
        source = OpenCVFrameSource(
            "video",
            "virtual.mp4",
            capture_factory=lambda _: capture,
            monotonic_clock=lambda: 5.0,
            wall_clock=lambda: 10.0,
        )

        first = source.read()
        second = source.read()
        end = source.read()

        self.assertEqual(first.sequence_timestamp, 0.0)
        self.assertAlmostEqual(second.sequence_timestamp, 0.05)
        self.assertTrue(end.eof)
        self.assertFalse(end.valid)
        self.assertTrue(source.reset())
        looped = source.read()
        self.assertTrue(looped.reset)
        self.assertEqual(looped.sequence_timestamp, 0.0)

    def test_default_cli_is_one_external_camera(self):
        args = parse_args([])
        self.assertEqual(args.mode, "camera")
        self.assertIsNone(args.source)
        self.assertIn("UNAVAILABLE", CABIN_STATUS)


class _FakeDetector:
    def __init__(self, detections=None):
        self.calls = 0
        self.detections = detections

    def process(self, frame):
        self.calls += 1
        if self.detections is not None:
            return list(self.detections)
        return [{"box": [0, 0, 2, 2], "distance": 10.0}]


class _FakeDistance:
    def calculate(self, detections):
        return detections


class _FakeCollision:
    def __init__(self):
        self.timestamps = []
        self.reset_calls = 0

    def process(self, detections, frame_width=None, timestamp=None):
        self.timestamps.append(timestamp)
        return detections

    def reset(self):
        self.reset_calls += 1


class _FakeFusion:
    def __init__(self):
        self.calls = []
        self.reset_calls = 0

    def evaluate(self, vehicle_data=None, face_data=None):
        self.calls.append((vehicle_data, face_data))
        return FusionResult(ext_score=0.4, fused_score=0.14)

    def reset(self):
        self.reset_calls += 1


class _FakePlanner:
    def __init__(self, *, ready=True):
        self.calls = []
        self.model_status = "ready" if ready else "fallback (test)"
        self.rssm = SimpleNamespace(ready=ready, observe_count=1) if ready else None
        self.reset_calls = 0

    def plan(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            action="KEEP",
            target_decel=0.0,
            prediction_source="rssm_hybrid" if kwargs["external_perception_valid"] else "kinematic",
            model_uncertainty=0.1,
            predicted_risk=0.2,
        )

    def reset(self):
        self.reset_calls += 1
        if self.rssm is not None:
            self.rssm.observe_count = 0


class _FakeLogger:
    def __init__(self):
        self.calls = []

    def log(self, decision, timestamp=None):
        self.calls.append((decision, timestamp))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.detector = _FakeDetector()
        self.collision = _FakeCollision()
        self.fusion = _FakeFusion()
        self.planner = _FakePlanner()
        self.logger = _FakeLogger()
        self.pipeline = ExternalOnlyPipeline(
            self.detector,
            _FakeDistance(),
            self.collision,
            self.fusion,
            self.planner,
            self.logger,
        )

    def test_valid_frame_is_external_only_open_loop_with_source_time(self):
        packet = FramePacket(
            np.zeros((3, 5, 3), np.uint8),
            wall_timestamp=100.0,
            sequence_timestamp=2.5,
            frame_id=7,
            source_kind="camera",
        )

        _, _, decision = self.pipeline.process(packet)

        self.assertEqual(decision.prediction_source, "rssm_hybrid")
        self.assertEqual(self.collision.timestamps, [2.5])
        self.assertEqual(self.fusion.calls[0][1], None)
        call = self.planner.calls[0]
        self.assertIsNone(call["face_data"])
        self.assertIsNone(call["applied_action"])
        self.assertTrue(call["external_perception_valid"])
        self.assertEqual(call["sequence_timestamp"], 2.5)

    def test_fresh_frame_with_zero_detections_remains_valid(self):
        detector = _FakeDetector(detections=[])
        pipeline = ExternalOnlyPipeline(
            detector,
            _FakeDistance(),
            self.collision,
            self.fusion,
            self.planner,
            self.logger,
        )
        packet = FramePacket(
            np.zeros((3, 5, 3), np.uint8), 100.0, 2.5, 7, "camera"
        )

        _, vehicles, _ = pipeline.process(packet)

        self.assertEqual(vehicles, [])
        self.assertEqual(self.fusion.calls[-1], ([], None))
        self.assertTrue(self.planner.calls[-1]["external_perception_valid"])

    def test_packet_reset_resets_temporal_pipeline_once(self):
        packet = FramePacket(
            np.zeros((3, 5, 3), np.uint8),
            100.0,
            0.0,
            0,
            "video",
            reset=True,
        )

        self.pipeline.process(packet)

        self.assertEqual(self.collision.reset_calls, 1)
        self.assertEqual(self.fusion.reset_calls, 1)
        self.assertEqual(self.planner.reset_calls, 1)

    def test_source_loss_is_not_a_valid_empty_road(self):
        valid = FramePacket(
            np.zeros((3, 5, 3), np.uint8), 100.0, 2.5, 7, "camera"
        )
        lost = FramePacket(
            None,
            100.8,
            3.3,
            8,
            "camera",
            valid=False,
            perception_age_sec=0.8,
        )
        self.pipeline.process(valid)
        self.pipeline.process(lost)

        self.assertEqual(self.detector.calls, 1)
        self.assertEqual(len(self.fusion.calls), 1)
        call = self.planner.calls[-1]
        self.assertFalse(call["external_perception_valid"])
        self.assertEqual(call["external_perception_age_sec"], 0.8)
        self.assertEqual(call["vehicle_data"], [])
        self.assertIsNone(call["applied_action"])

    def test_strict_validation_rejects_unready_planner(self):
        require_ready_planner(self.planner, allow_fallback=False)
        with self.assertRaisesRegex(RuntimeError, "requires a ready checkpoint"):
            require_ready_planner(_FakePlanner(ready=False), allow_fallback=False)
        require_ready_planner(_FakePlanner(ready=False), allow_fallback=True)

    def test_stats_distinguish_expected_sensor_loss_from_valid_fallback(self):
        stats = ValidationStats()
        rssm = self.planner.plan(external_perception_valid=True)
        lost = self.planner.plan(external_perception_valid=False)
        stats.record(
            rssm,
            valid=True,
            rssm_observe_count=1,
            sequence_timestamp=0.0,
        )
        stats.record(lost, valid=False)
        summary = stats.as_dict(planner=self.planner, source_description="fake")
        self.assertEqual(summary["rssm_frames"], 1)
        self.assertEqual(summary["fallback_valid_frames"], 0)
        self.assertEqual(summary["sensor_loss_events"], 1)
        self.assertEqual(summary["control_mode"], "open_loop_advisory")

    @staticmethod
    def _decision(source="rssm_hybrid"):
        return SimpleNamespace(
            action="KEEP",
            prediction_source=source,
            model_uncertainty=0.1,
        )

    def test_strict_validation_rejects_empty_run(self):
        errors = strict_validation_errors(ValidationStats())
        self.assertTrue(any("no fresh" in error for error in errors))
        self.assertTrue(any("no valid frame used" in error for error in errors))

    def test_strict_validation_rejects_single_posterior(self):
        stats = ValidationStats()
        stats.record(
            self._decision(),
            valid=True,
            rssm_observe_count=1,
            sequence_timestamp=0.0,
        )
        errors = strict_validation_errors(stats)
        self.assertTrue(any("posterior updates" in error for error in errors))
        self.assertTrue(any("source-time span" in error for error in errors))

    def test_strict_validation_rejects_repeated_reanchors(self):
        stats = ValidationStats()
        for timestamp in (0.0, 0.5, 1.0):
            stats.record(
                self._decision(),
                valid=True,
                rssm_observe_count=1,
                sequence_timestamp=timestamp,
            )
        errors = strict_validation_errors(stats)
        self.assertTrue(any("posterior updates" in error for error in errors))
        self.assertFalse(any("source-time span" in error for error in errors))

    def test_strict_validation_rejects_any_valid_fallback(self):
        stats = ValidationStats()
        for index, timestamp in enumerate((0.0, 0.25, 0.5, 0.75)):
            stats.record(
                self._decision("kinematic" if index == 2 else "rssm_hybrid"),
                valid=True,
                rssm_observe_count=index + 1,
                sequence_timestamp=timestamp,
            )
        errors = strict_validation_errors(stats)
        self.assertTrue(any("non-RSSM" in error for error in errors))

    def test_strict_validation_accepts_contiguous_chain(self):
        stats = ValidationStats()
        for index, timestamp in enumerate((0.0, 0.25, 0.5, 0.75)):
            stats.record(
                self._decision(),
                valid=True,
                rssm_observe_count=index + 1,
                sequence_timestamp=timestamp,
            )
        self.assertEqual(strict_validation_errors(stats), [])


    def test_strict_outcome_reports_rssm_validation_success(self):
        stats = ValidationStats()
        for index, timestamp in enumerate((0.0, 0.25, 0.5, 0.75)):
            stats.record(
                self._decision(),
                valid=True,
                rssm_observe_count=index + 1,
                sequence_timestamp=timestamp,
            )

        outcome, errors = build_validation_outcome(
            stats,
            allow_kinematic_fallback=False,
            min_posterior_updates=4,
            min_source_span_sec=0.75,
            runtime_exit_code=0,
        )

        self.assertEqual(errors, [])
        self.assertTrue(outcome["run_completed"])
        self.assertTrue(outcome["rssm_validation_passed"])
        self.assertTrue(outcome["validation_passed"])
        self.assertIsNone(outcome["diagnostic_passed"])

    def test_fallback_diagnostic_never_claims_rssm_validation(self):
        stats = ValidationStats()
        stats.record(
            self._decision("kinematic"),
            valid=True,
            sequence_timestamp=0.0,
        )

        outcome, errors = build_validation_outcome(
            stats,
            allow_kinematic_fallback=True,
            min_posterior_updates=4,
            min_source_span_sec=0.75,
            runtime_exit_code=0,
        )

        self.assertEqual(errors, [])
        self.assertTrue(outcome["run_completed"])
        self.assertTrue(outcome["diagnostic_passed"])
        self.assertFalse(outcome["rssm_validation_passed"])
        self.assertFalse(outcome["validation_passed"])

    def test_validation_outcome_rejects_empty_diagnostic_and_runtime_error(self):
        empty_outcome, empty_errors = build_validation_outcome(
            ValidationStats(),
            allow_kinematic_fallback=True,
            min_posterior_updates=4,
            min_source_span_sec=0.75,
            runtime_exit_code=0,
        )
        self.assertTrue(empty_errors)
        self.assertFalse(empty_outcome["run_completed"])
        self.assertFalse(empty_outcome["diagnostic_passed"])

        stats = ValidationStats()
        for index, timestamp in enumerate((0.0, 0.25, 0.5, 0.75)):
            stats.record(
                self._decision(),
                valid=True,
                rssm_observe_count=index + 1,
                sequence_timestamp=timestamp,
            )
        failed_outcome, failed_errors = build_validation_outcome(
            stats,
            allow_kinematic_fallback=False,
            min_posterior_updates=4,
            min_source_span_sec=0.75,
            runtime_exit_code=2,
        )
        self.assertTrue(any("exit code 2" in error for error in failed_errors))
        self.assertFalse(failed_outcome["run_completed"])
        self.assertFalse(failed_outcome["rssm_validation_passed"])
        self.assertFalse(failed_outcome["validation_passed"])


class CollisionSourceTimeTests(unittest.TestCase):
    @staticmethod
    def _detection(distance):
        return {
            "box": [100, 100, 200, 200],
            "class_id": 2,
            "distance": distance,
        }

    def test_collision_warner_uses_explicit_source_time_and_ignores_duplicate(self):
        warner = CollisionWarner({"cooldown_sec": 0.0})
        warner.process([self._detection(10.0)], frame_width=640, timestamp=0.0)
        second = warner.process(
            [self._detection(9.0)], frame_width=640, timestamp=0.25
        )[0]
        duplicate = warner.process(
            [self._detection(1.0)], frame_width=640, timestamp=0.25
        )[0]
        third = warner.process(
            [self._detection(8.0)], frame_width=640, timestamp=0.50
        )[0]

        self.assertAlmostEqual(second["rel_speed"], 1.6)
        self.assertEqual(duplicate["rel_speed"], 0.0)
        self.assertAlmostEqual(third["rel_speed"], 2.56)
        with self.assertRaisesRegex(ValueError, "finite"):
            warner.process([], timestamp=float("nan"))


class _FakeCarlaSensor:
    def __init__(self):
        self.callback = None
        self.stop_calls = 0
        self.destroy_calls = 0
        self.id = 42

    def listen(self, callback):
        self.callback = callback

    def emit(self, image):
        self.callback(image)

    def stop(self):
        self.stop_calls += 1

    def destroy(self):
        self.destroy_calls += 1


def _carla_image(frame, timestamp, bgra):
    array = np.asarray(bgra, dtype=np.uint8)
    return SimpleNamespace(
        frame=frame,
        timestamp=timestamp,
        width=array.shape[1],
        height=array.shape[0],
        raw_data=array.tobytes(),
    )


class CarlaAdapterTests(unittest.TestCase):
    def test_bgra_conversion_copies_bgr_channels(self):
        image = _carla_image(1, 0.1, [[[1, 2, 3, 255], [4, 5, 6, 255]]])
        frame = carla_image_to_bgr(image)
        self.assertEqual(frame.shape, (1, 2, 3))
        self.assertEqual(frame.tolist(), [[[1, 2, 3], [4, 5, 6]]])
        self.assertTrue(frame.flags["OWNDATA"])

    def test_carla_queue_keeps_latest_frame_and_close_is_idempotent(self):
        sensor = _FakeCarlaSensor()
        source = CarlaSensorSource(sensor, timeout_sec=0.01, owns_sensor=True)
        sensor.emit(_carla_image(10, 1.0, [[[1, 2, 3, 4]]]))
        sensor.emit(_carla_image(11, 1.1, [[[5, 6, 7, 8]]]))

        packet = source.read()

        self.assertEqual(packet.frame_id, 11)
        self.assertEqual(packet.sequence_timestamp, 1.1)
        self.assertEqual(packet.frame.tolist(), [[[5, 6, 7]]])
        self.assertTrue(packet.reset)
        sensor.emit(_carla_image(12, 1.2, [[[9, 10, 11, 12]]]))
        self.assertFalse(source.read().reset)
        source.close()
        source.close()
        self.assertEqual(sensor.stop_calls, 1)
        self.assertEqual(sensor.destroy_calls, 1)

    def test_carla_reset_is_consumed_once_across_pop_callback_race(self):
        sensor = _FakeCarlaSensor()
        source = CarlaSensorSource(sensor, timeout_sec=1.0)
        sensor.emit(_carla_image(1, 0.1, [[[1, 2, 3, 4]]]))

        popped = threading.Event()
        resume = threading.Event()
        original_get = source._queue.get

        def controlled_get(*args, **kwargs):
            item = original_get(*args, **kwargs)
            popped.set()
            if not resume.wait(1.0):
                raise RuntimeError("test did not resume queue consumer")
            return item

        source._queue.get = controlled_get
        packets = []
        errors = []

        def consume_first():
            try:
                packets.append(source.read())
            except Exception as exc:
                errors.append(exc)

        reader = threading.Thread(target=consume_first)
        reader.start()
        self.assertTrue(popped.wait(1.0))
        sensor.emit(_carla_image(2, 0.2, [[[5, 6, 7, 8]]]))
        resume.set()
        reader.join(1.0)
        source._queue.get = original_get

        self.assertFalse(reader.is_alive())
        if errors:
            raise errors[0]
        self.assertEqual(len(packets), 1)
        self.assertTrue(packets[0].reset)
        second = source.read()
        self.assertEqual(second.frame_id, 2)
        self.assertFalse(second.reset)
        source.close()

    def test_new_episode_reset_survives_latest_frame_replacement(self):
        sensor = _FakeCarlaSensor()
        source = CarlaSensorSource(sensor, timeout_sec=0.01, episode_id="one")
        sensor.emit(_carla_image(10, 1.0, [[[1, 2, 3, 4]]]))
        sensor.emit(_carla_image(11, 1.1, [[[5, 6, 7, 8]]]))
        first = source.read()
        self.assertTrue(first.reset)
        self.assertEqual(first.episode_id, "one")

        source.begin_episode("two")
        sensor.emit(_carla_image(1, 0.1, [[[9, 10, 11, 12]]]))
        sensor.emit(_carla_image(2, 0.2, [[[13, 14, 15, 16]]]))
        second = source.read()

        self.assertTrue(second.reset)
        self.assertEqual(second.frame_id, 2)
        self.assertEqual(second.episode_id, "two")
        source.close()

    def test_carla_timeout_is_invalid_not_empty_road(self):
        sensor = _FakeCarlaSensor()
        source = CarlaSensorSource(sensor, timeout_sec=0.01)
        started = time.monotonic()
        packet = source.read()
        self.assertGreaterEqual(time.monotonic() - started, 0.005)
        self.assertFalse(packet.valid)
        self.assertFalse(packet.eof)
        self.assertIsNone(packet.frame)
        self.assertTrue(packet.reset)

        same_episode_timeout = source.read()
        self.assertFalse(same_episode_timeout.reset)

        source.begin_episode("next")
        next_timeout = source.read()
        self.assertTrue(next_timeout.reset)
        self.assertEqual(next_timeout.episode_id, "next")
        self.assertFalse(source.read().reset)
        source.close()
        self.assertFalse(source.reset())

    def test_carla_rejects_a_queued_frame_that_became_stale(self):
        sensor = _FakeCarlaSensor()
        monotonic_now = [10.0]
        source = CarlaSensorSource(
            sensor,
            timeout_sec=0.01,
            monotonic_clock=lambda: monotonic_now[0],
        )
        sensor.emit(_carla_image(1, 0.1, [[[1, 2, 3, 4]]]))
        monotonic_now[0] = 10.02

        packet = source.read()

        self.assertFalse(packet.valid)
        self.assertIsNone(packet.frame)
        self.assertTrue(packet.reset)
        self.assertAlmostEqual(packet.perception_age_sec, 0.02)
        monotonic_now[0] = 10.05
        next_timeout = source.read()
        self.assertFalse(next_timeout.reset)
        self.assertAlmostEqual(next_timeout.perception_age_sec, 0.05)
        source.close()

    def test_carla_loss_age_accumulates_before_any_valid_frame(self):
        sensor = _FakeCarlaSensor()
        monotonic_now = [20.0]
        source = CarlaSensorSource(
            sensor,
            timeout_sec=0.001,
            monotonic_clock=lambda: monotonic_now[0],
        )

        first = source.read()
        monotonic_now[0] = 20.2
        second = source.read()

        self.assertEqual(first.perception_age_sec, 0.0)
        self.assertAlmostEqual(second.perception_age_sec, 0.2)
        source.close()

    def test_carla_owned_sensor_is_destroyed_even_when_stop_raises(self):
        class StopFailureSensor(_FakeCarlaSensor):
            def stop(self):
                super().stop()
                raise RuntimeError("stop failed")

        sensor = StopFailureSensor()
        source = CarlaSensorSource(sensor, timeout_sec=0.01, owns_sensor=True)

        with self.assertRaisesRegex(RuntimeError, "stop failed"):
            source.close()

        self.assertEqual(sensor.destroy_calls, 1)
        source.close()


if __name__ == "__main__":
    unittest.main()

