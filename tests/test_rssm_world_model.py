"""Fast CPU contract tests for the Dreamer-style RSSM world model."""

from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch
from collections.abc import Mapping
from pathlib import Path

import torch

from src.core.mrm_planner import CandidateActions, WorldState
from src.core.rssm_world_model import (
    ActionCodec,
    RSSMConfig,
    RSSMInferenceEngine,
    TinyRSSM,
    WorldStateCodec,
    checkpoint_sha256,
)


def _field(value: object, name: str):
    if isinstance(value, Mapping):
        return value[name]
    return getattr(value, name)


class _RecordingTinyRSSM(TinyRSSM):
    """TinyRSSM test double that records observed and imagined actions."""

    def __init__(self, config: RSSMConfig):
        super().__init__(config)
        self.observed_actions: list[torch.Tensor] = []
        self.imagined_actions: list[torch.Tensor] = []

    def observe_step(self, previous, action, observation, sample=True):
        self.observed_actions.append(action.detach().cpu().clone())
        return super().observe_step(previous, action, observation, sample=sample)

    def imagine_step(self, previous, action, sample=True):
        self.imagined_actions.append(action.detach().cpu().clone())
        return super().imagine_step(previous, action, sample=sample)


class CodecTests(unittest.TestCase):
    def test_world_state_codec_shape_range_and_missing_lead_sentinel(self) -> None:
        codec = WorldStateCodec()
        self.assertEqual(len(codec.OBS_FIELDS), 13)
        self.assertEqual(
            tuple(codec.OBS_FIELDS),
            (
                "has_lead_vehicle",
                "lead_distance",
                "lead_ttc",
                "closing_speed",
                "lane_relevance",
                "lead_warning_level",
                "ext_score",
                "int_score",
                "cross_score",
                "fused_score",
                "fused_level",
                "fatigue_score",
                "attention_score",
            ),
        )

        missing = WorldState(
            has_lead_vehicle=False,
            lead_distance=99.0,
            lead_ttc=99.0,
            closing_speed=0.0,
        )
        encoded = codec.encode(missing)
        self.assertEqual(encoded.shape, (13,))
        self.assertEqual(encoded.device.type, "cpu")
        self.assertTrue(torch.isfinite(encoded).all().item())
        self.assertTrue(((encoded >= 0.0) & (encoded <= 1.0)).all().item())

        distance_index = codec.OBS_FIELDS.index("lead_distance")
        ttc_index = codec.OBS_FIELDS.index("lead_ttc")
        self.assertAlmostEqual(float(encoded[distance_index]), 0.99, places=6)
        self.assertEqual(float(encoded[ttc_index]), 1.0)

        decoded = codec.decode_geometry(encoded)
        self.assertFalse(bool(_field(decoded, "has_lead")))
        self.assertEqual(float(_field(decoded, "distance")), 99.0)
        self.assertEqual(float(_field(decoded, "ttc")), 99.0)

        extreme = WorldState(
            has_lead_vehicle=True,
            lead_distance=-5.0,
            lead_ttc=500.0,
            closing_speed=100.0,
            lane_relevance=-2.0,
            lead_warning_level=99,
            ext_score=-1.0,
            int_score=2.0,
            cross_score=3.0,
            fused_score=-4.0,
            fused_level=99,
            fatigue_score=8.0,
            attention_score=-8.0,
        )
        clipped = codec.encode(extreme)
        self.assertTrue(torch.isfinite(clipped).all().item())
        self.assertTrue(((clipped >= 0.0) & (clipped <= 1.0)).all().item())

        batch = codec.encode_batch([missing, extreme])
        self.assertEqual(batch.shape, (2, 13))
        self.assertTrue(torch.isfinite(batch).all().item())

    def test_action_codec_shape_round_trip_and_invalid_action(self) -> None:
        codec = ActionCodec(max_decel=8.0, max_delay=0.5)
        action = CandidateActions().generate(WorldState())[2]
        encoded = codec.encode(action)
        self.assertEqual(encoded.shape, (3,))
        self.assertTrue(torch.isfinite(encoded).all().item())
        self.assertTrue(((encoded >= 0.0) & (encoded <= 1.0)).all().item())

        self.assertAlmostEqual(float(encoded[0]), action.target_decel / 8.0, places=5)
        self.assertAlmostEqual(float(encoded[1]), action.response_delay_sec / 0.5, places=5)
        self.assertEqual(float(encoded[2]), 1.0)

        invalid = codec.encode(None)
        self.assertEqual(invalid.shape, (3,))
        self.assertTrue(torch.equal(invalid, torch.zeros(3)))

        with self.assertRaises(ValueError):
            codec.encode({"name": "BRAKE"})


class TinyRSSMTests(unittest.TestCase):
    @staticmethod
    def _config() -> RSSMConfig:
        return RSSMConfig(
            obs_dim=13,
            action_dim=3,
            deter_dim=16,
            stoch_dim=2,
            classes=4,
            embed_dim=16,
            hidden_dim=16,
            dt_sec=0.25,
            samples=2,
        )

    def test_latent_shapes_finite_loss_and_gradients(self) -> None:
        torch.manual_seed(7)
        config = self._config()
        model = TinyRSSM(config).cpu()

        initial = model.initial(batch_size=2, device=torch.device("cpu"))
        self.assertEqual(initial.deter.shape, (2, config.deter_dim))
        self.assertEqual(
            initial.stoch.shape,
            (2, config.stoch_dim, config.classes),
        )
        self.assertTrue(torch.isfinite(initial.deter).all().item())
        self.assertTrue(torch.isfinite(initial.stoch).all().item())

        observations = torch.rand(2, 4, config.obs_dim)
        actions = torch.rand(2, 4, config.action_dim)
        actions[..., -1] = 1.0
        risk_targets = torch.rand(2, 4)
        continues = torch.ones(2, 4)

        losses = model.loss_sequence(
            observations=observations,
            actions=actions,
            risk_targets=risk_targets,
            continues=continues,
        )
        self.assertIsInstance(losses, Mapping)
        required = {
            "loss",
            "total_loss",
            "obs_loss",
            "risk_loss",
            "continue_loss",
            "dyn_loss",
            "rep_loss",
        }
        self.assertTrue(required.issubset(losses.keys()))
        for name in required:
            value = losses[name]
            self.assertTrue(torch.is_tensor(value), name)
            self.assertEqual(value.numel(), 1, name)
            self.assertTrue(torch.isfinite(value).all().item(), name)

        loss = losses["loss"]
        self.assertTrue(loss.requires_grad)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(grad).all().item() for grad in gradients))
        self.assertTrue(any(torch.count_nonzero(grad).item() > 0 for grad in gradients))

    def test_config_rejects_fractional_or_nan_samples(self) -> None:
        for value in (1.5, float("nan"), 2**62):
            with self.subTest(samples=value):
                with self.assertRaises(ValueError):
                    RSSMConfig.from_mapping({"samples": value})

    def test_single_step_default_risk_is_finite_and_mask_shape_is_strict(self) -> None:
        torch.manual_seed(9)
        model = TinyRSSM(self._config()).cpu()
        observations = torch.rand(2, 1, model.config.obs_dim)
        actions = torch.zeros(2, 1, model.config.action_dim)

        losses = model.loss_sequence(observations=observations, actions=actions)
        self.assertTrue(
            all(torch.isfinite(value).all().item() for value in losses.values())
        )

        with self.assertRaises(ValueError):
            model.loss_sequence(
                observations=observations,
                actions=actions,
                valid_mask=torch.ones(2, 1, 1),
            )

    def test_mid_sequence_is_first_zeros_only_reset_rows_action(self) -> None:
        torch.manual_seed(10)
        model = _RecordingTinyRSSM(self._config()).cpu().eval()
        observations = torch.rand(2, 4, model.config.obs_dim)
        actions = torch.tensor(
            [
                [
                    [0.1, 0.2, 1.0],
                    [0.2, 0.3, 1.0],
                    [0.3, 0.4, 1.0],
                    [0.4, 0.5, 1.0],
                ],
                [
                    [0.5, 0.4, 1.0],
                    [0.4, 0.3, 1.0],
                    [0.3, 0.2, 1.0],
                    [0.2, 0.1, 1.0],
                ],
            ],
            dtype=torch.float32,
        )
        original_actions = actions.clone()
        is_first = torch.zeros(2, 4, dtype=torch.bool)
        is_first[0, 2] = True

        model.observe_sequence(
            observations=observations,
            actions=actions,
            is_first=is_first,
            sample=False,
        )

        self.assertEqual(len(model.observed_actions), 4)
        for step in (0, 1, 3):
            self.assertTrue(torch.equal(model.observed_actions[step], actions[:, step]))
        self.assertTrue(torch.equal(model.observed_actions[2][0], torch.zeros(3)))
        self.assertTrue(torch.equal(model.observed_actions[2][1], actions[1, 2]))
        self.assertTrue(torch.equal(actions, original_actions))

    def test_checkpoint_round_trip_and_strict_failures(self) -> None:
        torch.manual_seed(11)
        model = TinyRSSM(self._config()).cpu()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "rssm.pt"
            model.save_checkpoint(checkpoint, metrics={"unit_test": True})

            digest = checkpoint_sha256(checkpoint)
            restored, metrics = TinyRSSM.load_checkpoint(
                checkpoint,
                device="cpu",
                expected_sha256=digest,
            )
            self.assertIsInstance(restored, TinyRSSM)
            self.assertEqual(metrics, {"unit_test": True})
            for key, expected in model.state_dict().items():
                self.assertTrue(torch.equal(restored.state_dict()[key], expected), key)

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                TinyRSSM.load_checkpoint(
                    checkpoint,
                    device="cpu",
                    expected_sha256="0" * 64,
                )

            with patch(
                "src.core.rssm_world_model.MAX_CHECKPOINT_BYTES",
                checkpoint.stat().st_size - 1,
            ), self.assertRaisesRegex(ValueError, "exceeds"):
                TinyRSSM.load_checkpoint(checkpoint, device="cpu")

            with self.assertRaises(FileNotFoundError):
                TinyRSSM.load_checkpoint(root / "missing.pt", device="cpu")

            malformed = root / "missing-schema.pt"
            torch.save({"model_state_dict": model.state_dict()}, malformed)
            with self.assertRaises(ValueError):
                TinyRSSM.load_checkpoint(malformed, device="cpu")

            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            payload["schema_version"] = 999
            bad_schema = root / "bad-schema.pt"
            torch.save(payload, bad_schema)
            with self.assertRaises(ValueError):
                TinyRSSM.load_checkpoint(bad_schema, device="cpu")


class InferenceEngineTests(unittest.TestCase):
    def test_cvar_uses_fractional_tail_boundary(self) -> None:
        config = replace(TinyRSSMTests._config(), cvar_alpha=0.50)
        engine = RSSMInferenceEngine.from_model(TinyRSSM(config).cpu().eval())
        values = torch.tensor([[1.0, 0.2, 0.0]])

        actual = engine._cvar(values)

        self.assertAlmostEqual(float(actual[0]), (1.0 + 0.5 * 0.2) / 1.5, places=6)

    def test_cvar_tiny_positive_tail_returns_worst_sample(self) -> None:
        config = replace(
            TinyRSSMTests._config(),
            cvar_alpha=math.nextafter(1.0, 0.0),
        )
        engine = RSSMInferenceEngine.from_model(TinyRSSM(config).cpu().eval())
        values = torch.tensor([[0.1, 0.3, 0.9, 0.2]])

        actual = engine._cvar(values)

        self.assertAlmostEqual(float(actual[0]), 0.9, places=6)

    def test_geometry_tail_uses_conservative_lower_order_statistic(self) -> None:
        model = _RecordingTinyRSSM(TinyRSSMTests._config()).cpu().eval()

        def decode(state):
            count = state.deter.shape[0]
            values = torch.zeros(count, model.config.obs_dim, device=state.deter.device)
            is_first_sample = torch.arange(count, device=state.deter.device) % 2 == 0
            values[:, 0] = 1.0
            values[:, 1] = torch.where(is_first_sample, 0.01, 0.99)
            values[:, 2] = torch.where(is_first_sample, 0.01, 0.99)
            return values

        model.decode_observation = decode
        model.predict_continue = lambda state: torch.ones(
            state.deter.shape[0], device=state.deter.device
        )
        engine = RSSMInferenceEngine.from_model(model)
        state = WorldState(has_lead_vehicle=True, lead_distance=50.0, lead_ttc=10.0)
        candidates = CandidateActions().generate(state)
        self.assertTrue(engine.observe(state, applied_action=None, timestamp=0.0))

        forecasts = engine.predict_many(candidates, horizons_sec=[0.25])

        for forecast in forecasts.values():
            self.assertAlmostEqual(forecast.min_distance, 1.0, places=5)
            self.assertAlmostEqual(forecast.trajectory[0]["distance"], 1.0, places=5)

    def test_imagination_rejects_unbounded_horizon_before_rollout(self) -> None:
        model = _RecordingTinyRSSM(TinyRSSMTests._config()).cpu().eval()
        engine = RSSMInferenceEngine.from_model(model)
        state = WorldState(has_lead_vehicle=False)
        candidates = CandidateActions().generate(state)
        self.assertTrue(engine.observe(state, applied_action=None, timestamp=0.0))
        model.imagined_actions.clear()

        with self.assertRaises(ValueError):
            engine.predict_many(candidates, horizons_sec=[1e308])
        self.assertEqual(len(model.imagined_actions), 0)

    def test_reset_observe_and_four_candidate_predictions(self) -> None:
        torch.manual_seed(19)
        model = TinyRSSM(TinyRSSMTests._config()).cpu().eval()
        engine = RSSMInferenceEngine.from_model(model)
        state = WorldState(
            timestamp=10.0,
            has_lead_vehicle=True,
            lead_distance=18.0,
            lead_ttc=2.4,
            closing_speed=7.5,
            lane_relevance=1.0,
            lead_warning_level=2,
            ext_score=0.72,
            int_score=0.35,
            cross_score=0.25,
            fused_score=0.61,
            fused_level=2,
            fatigue_score=0.30,
            attention_score=0.40,
        )
        candidates = CandidateActions().generate(state)
        self.assertEqual(len(candidates), 4)

        self.assertTrue(engine.observe(state, applied_action=None, timestamp=10.0))
        predictions = engine.predict_many(candidates, horizons_sec=[0.25, 0.50])
        self.assertEqual(len(predictions), 4)
        self.assertEqual(
            list(predictions),
            [action.name for action in candidates],
        )
        for prediction in predictions.values():
            for name in (
                "peak_risk",
                "terminal_risk",
                "min_distance",
                "min_ttc",
                "uncertainty",
            ):
                self.assertTrue(math.isfinite(float(_field(prediction, name))), name)
            self.assertGreaterEqual(float(_field(prediction, "peak_risk")), 0.0)
            self.assertLessEqual(float(_field(prediction, "peak_risk")), 1.0)
            self.assertGreaterEqual(float(_field(prediction, "terminal_risk")), 0.0)
            self.assertLessEqual(float(_field(prediction, "terminal_risk")), 1.0)
            self.assertEqual(len(_field(prediction, "trajectory")), 2)

        engine.reset()
        self.assertFalse(engine.has_belief)
        self.assertTrue(engine.observe(state, applied_action=None, timestamp=20.0))
        self.assertEqual(
            len(engine.predict_many(candidates, horizons_sec=[0.25])), 4
        )

    def test_observe_throttles_and_encodes_brake_onset_only_once(self) -> None:
        model = _RecordingTinyRSSM(TinyRSSMTests._config()).cpu().eval()
        engine = RSSMInferenceEngine.from_model(model)
        state = WorldState(
            has_lead_vehicle=True,
            lead_distance=16.0,
            lead_ttc=2.0,
        )
        brake = CandidateActions().generate(state)[2]

        self.assertTrue(engine.observe(state, applied_action=None, timestamp=0.0))
        self.assertTrue(engine.observe(state, applied_action=brake, timestamp=0.25))
        onset = model.observed_actions[-1][0]
        self.assertAlmostEqual(
            float(onset[1]),
            brake.response_delay_sec / engine.config.max_delay,
            places=6,
        )
        self.assertEqual(float(onset[2]), 1.0)

        count_before_throttle = engine.observe_count
        calls_before_throttle = len(model.observed_actions)
        self.assertFalse(engine.observe(state, applied_action=brake, timestamp=0.35))
        self.assertEqual(engine.observe_count, count_before_throttle)
        self.assertEqual(len(model.observed_actions), calls_before_throttle)

        self.assertTrue(engine.observe(state, applied_action=brake, timestamp=0.50))
        held = model.observed_actions[-1][0]
        self.assertEqual(float(held[1]), 0.0)
        self.assertEqual(float(held[2]), 1.0)

        changed_brake = {
            "name": brake.name,
            "target_decel": brake.target_decel,
            "response_delay_sec": brake.response_delay_sec,
            "action_changed": True,
        }
        self.assertTrue(
            engine.observe(state, applied_action=changed_brake, timestamp=0.75)
        )
        changed = model.observed_actions[-1][0]
        self.assertAlmostEqual(
            float(changed[1]),
            brake.response_delay_sec / engine.config.max_delay,
            places=6,
        )
        self.assertEqual(float(changed[2]), 1.0)

    def test_imagination_zeros_delay_only_for_currently_applied_brake(self) -> None:
        model = _RecordingTinyRSSM(TinyRSSMTests._config()).cpu().eval()
        engine = RSSMInferenceEngine.from_model(model)
        state = WorldState(
            has_lead_vehicle=True,
            lead_distance=15.0,
            lead_ttc=2.0,
        )
        candidates = CandidateActions().generate(state)
        brake = candidates[2]

        self.assertTrue(engine.observe(state, applied_action=None, timestamp=0.0))
        self.assertTrue(engine.observe(state, applied_action=brake, timestamp=0.25))
        model.imagined_actions.clear()

        predictions = engine.predict_many(candidates, horizons_sec=[0.25])
        self.assertEqual(len(predictions), 4)
        self.assertEqual(len(model.imagined_actions), 1)
        first_step = model.imagined_actions[0].reshape(
            len(candidates),
            model.config.samples,
            model.config.action_dim,
        )
        for index, candidate in enumerate(candidates):
            expected_delay = (
                0.0
                if candidate.name == brake.name
                else candidate.response_delay_sec / engine.config.max_delay
            )
            expected = torch.full(
                (model.config.samples,),
                expected_delay,
                dtype=first_step.dtype,
            )
            self.assertTrue(torch.allclose(first_step[index, :, 1], expected))
            self.assertTrue(torch.equal(first_step[index, :, 2], torch.ones_like(expected)))

    def test_imagination_preserves_current_remaining_delay(self) -> None:
        model = _RecordingTinyRSSM(TinyRSSMTests._config()).cpu().eval()
        engine = RSSMInferenceEngine.from_model(model)
        state = WorldState(has_lead_vehicle=True, lead_distance=15.0, lead_ttc=2.0)
        candidates = CandidateActions().generate(state)
        brake = candidates[2]

        self.assertTrue(engine.observe(state, applied_action=None, timestamp=0.0))
        receipt = {
            "name": brake.name,
            "target_decel": brake.target_decel,
            "remaining_delay_sec": 0.30,
            "action_changed": True,
        }
        self.assertTrue(engine.observe(state, applied_action=receipt, timestamp=0.25))
        model.imagined_actions.clear()

        engine.predict_many(candidates, horizons_sec=[0.75])
        self.assertEqual(len(model.imagined_actions), 3)
        for step, expected_delay in enumerate((0.30, 0.05, 0.0)):
            encoded = model.imagined_actions[step].reshape(
                len(candidates), model.config.samples, model.config.action_dim
            )
            expected = torch.full(
                (model.config.samples,),
                expected_delay / engine.config.max_delay,
                dtype=encoded.dtype,
            )
            self.assertTrue(torch.allclose(encoded[2, :, 1], expected))

    def test_same_name_with_different_setpoint_is_not_held(self) -> None:
        model = _RecordingTinyRSSM(TinyRSSMTests._config()).cpu().eval()
        engine = RSSMInferenceEngine.from_model(model)
        state = WorldState(has_lead_vehicle=True, lead_distance=15.0, lead_ttc=2.0)
        brake = CandidateActions().generate(state)[2]

        self.assertTrue(engine.observe(state, applied_action=None, timestamp=0.0))
        self.assertTrue(engine.observe(state, applied_action=brake, timestamp=0.25))
        mismatched = {
            "name": brake.name,
            "target_decel": 0.0,
            "response_delay_sec": brake.response_delay_sec,
            "action_changed": False,
        }
        self.assertTrue(engine.observe(state, applied_action=mismatched, timestamp=0.50))
        encoded = model.observed_actions[-1][0]
        self.assertAlmostEqual(
            float(encoded[1]),
            brake.response_delay_sec / engine.config.max_delay,
            places=6,
        )

    def test_low_continue_probability_is_treated_as_imagined_risk(self) -> None:
        model = _RecordingTinyRSSM(TinyRSSMTests._config()).cpu().eval()
        model.decode_observation = lambda state: torch.zeros(
            state.deter.shape[0], model.config.obs_dim, device=state.deter.device
        )
        model.predict_risk = lambda state: torch.zeros(
            state.deter.shape[0], device=state.deter.device
        )
        model.predict_continue = lambda state: torch.zeros(
            state.deter.shape[0], device=state.deter.device
        )
        engine = RSSMInferenceEngine.from_model(model)
        state = WorldState(has_lead_vehicle=False)
        candidates = CandidateActions().generate(state)

        self.assertTrue(engine.observe(state, applied_action=None, timestamp=0.0))
        forecasts = engine.predict_many(candidates, horizons_sec=[0.25])

        self.assertEqual(set(forecasts), {action.name for action in candidates})
        for forecast in forecasts.values():
            self.assertAlmostEqual(forecast.peak_risk, 1.0, places=6)
            self.assertAlmostEqual(forecast.terminal_risk, 1.0, places=6)

    def test_false_change_hint_cannot_hide_different_action_onset(self) -> None:
        model = _RecordingTinyRSSM(TinyRSSMTests._config()).cpu().eval()
        engine = RSSMInferenceEngine.from_model(model)
        state = WorldState(has_lead_vehicle=True, lead_distance=15.0, lead_ttc=2.0)
        slow_down, brake = CandidateActions().generate(state)[1:3]

        self.assertTrue(engine.observe(state, applied_action=None, timestamp=0.0))
        self.assertTrue(engine.observe(state, applied_action=brake, timestamp=0.25))
        changed_action = {
            "name": slow_down.name,
            "target_decel": slow_down.target_decel,
            "response_delay_sec": slow_down.response_delay_sec,
            "action_changed": False,
        }
        self.assertTrue(
            engine.observe(state, applied_action=changed_action, timestamp=0.50)
        )
        encoded = model.observed_actions[-1][0]
        self.assertAlmostEqual(
            float(encoded[0]),
            slow_down.target_decel / engine.config.max_decel,
            places=6,
        )
        self.assertAlmostEqual(
            float(encoded[1]),
            slow_down.response_delay_sec / engine.config.max_delay,
            places=6,
        )
        self.assertEqual(float(encoded[2]), 1.0)

    def test_duplicate_sensor_timestamp_is_not_reanchored(self) -> None:
        model = _RecordingTinyRSSM(TinyRSSMTests._config()).cpu().eval()
        engine = RSSMInferenceEngine.from_model(model)
        state = WorldState()

        self.assertTrue(engine.observe(state, applied_action=None, timestamp=5.0))
        first_belief = engine._belief
        self.assertFalse(engine.observe(state, applied_action=None, timestamp=5.0))

        self.assertIs(engine._belief, first_belief)
        self.assertEqual(engine.observe_count, 1)
        self.assertEqual(len(model.observed_actions), 1)
        self.assertFalse(engine._pending_action_change)

    def test_cadence_residual_tracks_long_term_dt_at_ten_hz(self) -> None:
        model = _RecordingTinyRSSM(TinyRSSMTests._config()).cpu().eval()
        engine = RSSMInferenceEngine.from_model(model)
        state = WorldState(has_lead_vehicle=False)
        accepted_times = []

        for index in range(31):
            timestamp = round(index * 0.10, 10)
            if engine.observe(state, applied_action=None, timestamp=timestamp):
                accepted_times.append(timestamp)

        self.assertEqual(len(accepted_times), 13)
        self.assertEqual(engine.observe_count, 13)
        intervals = {
            round(current - previous, 2)
            for previous, current in zip(accepted_times, accepted_times[1:])
        }
        self.assertEqual(intervals, {0.20, 0.30})

    def test_action_change_first_seen_on_accepted_call_reanchors_after_throttles(self) -> None:
        model = _RecordingTinyRSSM(TinyRSSMTests._config()).cpu().eval()
        engine = RSSMInferenceEngine.from_model(model)
        state = WorldState(has_lead_vehicle=True, lead_distance=14.0, lead_ttc=2.0)
        keep, brake = CandidateActions().generate(state)[0], CandidateActions().generate(state)[2]

        self.assertTrue(engine.observe(state, applied_action=keep, timestamp=0.0))
        self.assertFalse(engine.observe(state, applied_action=keep, timestamp=0.10))
        self.assertFalse(engine.observe(state, applied_action=keep, timestamp=0.20))
        changed = {
            "name": brake.name,
            "target_decel": brake.target_decel,
            "response_delay_sec": brake.response_delay_sec,
            "action_changed": True,
        }
        self.assertTrue(engine.observe(state, applied_action=changed, timestamp=0.30))

        self.assertEqual(engine.observe_count, 1)
        self.assertTrue(torch.equal(model.observed_actions[-1][0], torch.zeros(3)))

    def test_throttled_action_change_reanchors_next_accepted_observation(self) -> None:
        model = _RecordingTinyRSSM(TinyRSSMTests._config()).cpu().eval()
        engine = RSSMInferenceEngine.from_model(model)
        state = WorldState(has_lead_vehicle=True, lead_distance=14.0, lead_ttc=2.0)
        actions = CandidateActions().generate(state)
        slow_down, brake = actions[1], actions[2]

        self.assertTrue(engine.observe(state, applied_action=None, timestamp=0.0))
        self.assertTrue(engine.observe(state, applied_action=brake, timestamp=0.25))
        calls_before_throttle = len(model.observed_actions)
        self.assertFalse(
            engine.observe(state, applied_action=slow_down, timestamp=0.35)
        )
        self.assertEqual(len(model.observed_actions), calls_before_throttle)

        self.assertTrue(
            engine.observe(state, applied_action=slow_down, timestamp=0.50)
        )
        self.assertEqual(engine.observe_count, 1)
        reanchored_action = model.observed_actions[-1][0]
        self.assertEqual(float(reanchored_action[2]), 0.0)
        self.assertTrue(torch.equal(reanchored_action, torch.zeros(3)))

    def test_model_distribution_exception_trips_permanent_fallback(self) -> None:
        model = _RecordingTinyRSSM(TinyRSSMTests._config()).cpu().eval()
        with torch.no_grad():
            model.prior_net[-1].weight.fill_(torch.finfo(torch.float32).max)
            model.prior_net[-1].bias.fill_(torch.finfo(torch.float32).max)
        engine = RSSMInferenceEngine.from_model(model)
        state = WorldState(has_lead_vehicle=True, lead_distance=10.0, lead_ttc=1.0)

        self.assertFalse(engine.observe(state, applied_action=None, timestamp=0.0))
        first_call_count = len(model.observed_actions)
        self.assertFalse(engine.ready)
        self.assertIn("model disabled", engine.status)

        self.assertFalse(engine.observe(state, applied_action=None, timestamp=0.25))
        self.assertEqual(len(model.observed_actions), first_call_count)

    def test_nonfinite_model_output_trips_permanent_fallback(self) -> None:
        model = _RecordingTinyRSSM(TinyRSSMTests._config()).cpu().eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(1e30)
        engine = RSSMInferenceEngine.from_model(model)
        state = WorldState(has_lead_vehicle=True, lead_distance=10.0, lead_ttc=1.0)

        self.assertFalse(engine.observe(state, applied_action=None, timestamp=0.0))
        first_call_count = len(model.observed_actions)
        self.assertFalse(engine.ready)
        self.assertIn("model disabled", engine.status)

        self.assertFalse(engine.observe(state, applied_action=None, timestamp=0.25))
        self.assertEqual(len(model.observed_actions), first_call_count)

    def test_runtime_normalization_mismatch_raises_or_falls_back(self) -> None:
        model = TinyRSSM(TinyRSSMTests._config()).cpu().eval()
        mismatches = {"dt_sec": 0.5, "max_decel": 9.0}
        for field, value in mismatches.items():
            with self.subTest(source="model", field=field):
                with self.assertRaises(ValueError):
                    RSSMInferenceEngine.from_model(model, config={field: value})

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "rssm.pt"
            model.save_checkpoint(checkpoint)
            for field, value in mismatches.items():
                with self.subTest(source="checkpoint", field=field):
                    engine = RSSMInferenceEngine(
                        {
                            "enabled": True,
                            "device": "cpu",
                            "checkpoint": str(checkpoint),
                            "checkpoint_sha256": checkpoint_sha256(checkpoint),
                            field: value,
                        }
                    )
                    self.assertFalse(engine.ready)
                    self.assertIsNone(engine.model)
                    self.assertIn("fallback", engine.status)
                    self.assertIn(field, engine.status)

    def test_multistep_gap_resets_and_marks_latest_action_invalid(self) -> None:
        model = _RecordingTinyRSSM(TinyRSSMTests._config()).cpu().eval()
        engine = RSSMInferenceEngine.from_model(model)
        state = WorldState(
            has_lead_vehicle=True,
            lead_distance=12.0,
            lead_ttc=1.8,
        )
        brake = CandidateActions().generate(state)[2]

        self.assertTrue(engine.observe(state, applied_action=None, timestamp=0.0))
        self.assertTrue(engine.observe(state, applied_action=brake, timestamp=0.25))
        self.assertEqual(engine.observe_count, 2)

        self.assertTrue(engine.observe(state, applied_action=brake, timestamp=0.75))
        self.assertEqual(engine.observe_count, 1)
        reanchored_action = model.observed_actions[-1][0]
        self.assertEqual(float(reanchored_action[2]), 0.0)
        self.assertTrue(torch.equal(reanchored_action, torch.zeros(3)))

        self.assertTrue(engine.observe(state, applied_action=brake, timestamp=1.00))
        self.assertEqual(engine.observe_count, 2)
        held_after_reanchor = model.observed_actions[-1][0]
        self.assertEqual(float(held_after_reanchor[1]), 0.0)
        self.assertEqual(float(held_after_reanchor[2]), 1.0)


if __name__ == "__main__":
    unittest.main()
