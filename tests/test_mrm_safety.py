"""Fast safety-contract tests for the MRM planner and RSSM integration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.core.mrm_planner import (
    CandidateActions,
    MRMPlanner,
    RiskCost,
    RiskPredictor,
    SafetyShield,
    WorldState,
)
from src.core.risk_fusion import RiskFusionEngine
from src.core.rssm_world_model import (
    ACTION_CONTRACT_VERSION,
    RSSMConfig,
    TinyRSSM,
    build_fusion_feature_contract,
    checkpoint_sha256,
)


class PlannerConfigurationSafetyTests(unittest.TestCase):
    def test_risk_predictor_rejects_unbounded_horizon(self) -> None:
        for horizons in ([1e308], [float("inf")], "1.5"):
            with self.subTest(horizons=horizons), self.assertRaises(ValueError):
                RiskPredictor({"horizons_sec": horizons})

    def test_risk_predictor_sorts_and_deduplicates_horizons(self) -> None:
        predictor = RiskPredictor({"horizons_sec": [1.5, 0.5, 1.0, 0.5, 1.5]})
        self.assertEqual(predictor.horizons, [0.5, 1.0, 1.5])

    def test_risk_predictor_rejects_nonpositive_min_closing_speed(self) -> None:
        for value in (0.0, -0.01):
            with (
                self.subTest(min_closing_speed=value),
                self.assertRaisesRegex(ValueError, "must be positive"),
            ):
                RiskPredictor({"min_closing_speed": value})

    def test_risk_predictor_rejects_negative_ttc_or_distance_threshold(self) -> None:
        for name in ("ttc_critical", "ttc_safe", "dist_critical", "dist_safe"):
            with (
                self.subTest(threshold=name),
                self.assertRaisesRegex(ValueError, "thresholds must be non-negative"),
            ):
                RiskPredictor({name: -0.01})

    def test_risk_predictor_requires_safe_thresholds_above_critical(self) -> None:
        cases = (
            {"ttc_critical": 1.2, "ttc_safe": 1.2},
            {"ttc_critical": 1.2, "ttc_safe": 1.0},
            {"dist_critical": 3.0, "dist_safe": 3.0},
            {"dist_critical": 3.0, "dist_safe": 2.0},
        )
        for config in cases:
            with (
                self.subTest(config=config),
                self.assertRaisesRegex(ValueError, "must exceed critical thresholds"),
            ):
                RiskPredictor(config)

    def test_candidate_actions_reject_non_monotonic_decelerations(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increase"):
            CandidateActions({
                "decelerations": {
                    "KEEP": 0.0,
                    "SLOW_DOWN": 3.5,
                    "BRAKE": 3.0,
                    "EMERGENCY_BRAKE": 6.5,
                }
            })

    def test_candidate_actions_reject_decelerations_outside_model_range(self) -> None:
        cases = (
            {
                "decelerations": {"SLOW_DOWN": -0.1},
            },
            {
                "rssm": {"max_decel": 6.0},
            },
        )
        for config in cases:
            with self.subTest(config=config), self.assertRaises(ValueError):
                CandidateActions(config)

    def test_hybrid_candidate_actions_reject_delay_larger_than_model_step(self) -> None:
        with self.assertRaisesRegex(ValueError, "delays <= rssm.dt_sec"):
            CandidateActions({
                "predictor": "hybrid",
                "rssm": {"dt_sec": 0.10, "max_delay": 0.5},
            })

    def test_candidate_actions_reject_negative_or_decreasing_comfort_costs(self) -> None:
        cases = (
            ({"brake_cost": -0.01}, "non-negative"),
            (
                {"slow_down_cost": 0.10, "brake_cost": 0.05},
                "must not decrease",
            ),
        )
        for config, expected_message in cases:
            with (
                self.subTest(config=config),
                self.assertRaisesRegex(ValueError, expected_message),
            ):
                CandidateActions(config)

class VehicleSentinelFilteringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = MRMPlanner({"predictor": "kinematic"})
        self.invalid_detection = {
            "distance": -1.0,
            "ttc": -1.0,
            "rel_speed": 0.0,
            "lane_relevance": 1.0,
            "warning_level": 3,
        }

    def test_online_fusion_ignores_invalid_sentinel_detection(self) -> None:
        score, details = RiskFusionEngine({})._compute_ext_score(
            [self.invalid_detection]
        )
        self.assertEqual(score, 0.0)
        self.assertEqual(details["min_dist"], 99.0)
        self.assertEqual(details["min_ttc"], 99.0)
        self.assertEqual(details["max_level"], 0)

    def test_only_invalid_sentinel_detection_does_not_trigger_emergency_braking(self) -> None:
        result = self.planner.plan(
            SimpleNamespace(),
            vehicle_data=[self.invalid_detection],
            timestamp=1.0,
        )

        self.assertFalse(result.world_state.has_lead_vehicle)
        self.assertEqual(result.world_state.lead_distance, 99.0)
        self.assertEqual(result.world_state.lead_ttc, 99.0)
        self.assertEqual(result.action, "KEEP")

    def test_invalid_ttc_does_not_outrank_valid_approaching_object(self) -> None:
        state = self.planner.build_world_state(
            SimpleNamespace(),
            vehicle_data=[
                {
                    "distance": 100.0,
                    "ttc": -1.0,
                    "rel_speed": 0.0,
                    "lane_relevance": 1.0,
                    "warning_level": 0,
                },
                {
                    "distance": 20.0,
                    "ttc": 2.0,
                    "rel_speed": 1.0,
                    "lane_relevance": 1.0,
                    "warning_level": 0,
                },
            ],
            timestamp=1.0,
        )

        self.assertEqual(state.lead_distance, 20.0)
        self.assertEqual(state.lead_ttc, 2.0)
        self.assertEqual(state.closing_speed, 10.0)

    def test_ttc_implied_closing_speed_is_used_when_more_conservative(self) -> None:
        result = self.planner.plan(
            SimpleNamespace(),
            vehicle_data=[{
                "distance": 50.0,
                "ttc": 2.5,
                "rel_speed": 1.0,
                "lane_relevance": 1.0,
                "warning_level": 0,
            }],
            timestamp=1.0,
        )

        self.assertAlmostEqual(result.world_state.closing_speed, 20.0, places=6)
        self.assertEqual(result.action, "BRAKE")

    def test_mixed_detections_ignore_invalid_sentinel(self) -> None:
        valid_detection = {
            "distance": 20.0,
            "ttc": 4.0,
            "rel_speed": 5.0,
            "lane_relevance": 1.0,
            "warning_level": 0,
        }
        state = self.planner.build_world_state(
            SimpleNamespace(),
            vehicle_data=[self.invalid_detection, valid_detection],
            timestamp=1.0,
        )

        self.assertTrue(state.has_lead_vehicle)
        self.assertEqual(state.lead_distance, 20.0)
        self.assertEqual(state.lead_ttc, 4.0)
        self.assertEqual(state.closing_speed, 5.0)


class PerceptionLossSafetyTests(unittest.TestCase):
    @staticmethod
    def _rank(action: str) -> int:
        return {
            "KEEP": 0,
            "SLOW_DOWN": 1,
            "BRAKE": 2,
            "EMERGENCY_BRAKE": 3,
        }[action]

    def test_hazard_then_perception_loss_never_clears_the_road(self) -> None:
        planner = MRMPlanner({"predictor": "kinematic", "log_enable": False})
        vehicle = [{
            "distance": 1.0,
            "ttc": 0.5,
            "rel_speed": 2.0,
            "lane_relevance": 1.0,
            "warning_level": 2,
        }]
        danger = planner.plan(
            SimpleNamespace(),
            vehicle_data=vehicle,
            timestamp=1.0,
            external_perception_valid=True,
        )
        lost = planner.plan(
            SimpleNamespace(),
            vehicle_data=[],
            timestamp=1.1,
            external_perception_valid=False,
        )
        later = planner.plan(
            SimpleNamespace(),
            vehicle_data=[],
            timestamp=1.4,
            external_perception_valid=False,
        )

        self.assertEqual(danger.action, "EMERGENCY_BRAKE")
        self.assertGreaterEqual(self._rank(lost.action), self._rank(danger.action))
        self.assertTrue(lost.world_state.external_state_held)
        self.assertTrue(lost.world_state.has_lead_vehicle)
        self.assertLessEqual(
            later.world_state.lead_distance,
            lost.world_state.lead_distance,
        )
        self.assertLessEqual(later.world_state.lead_ttc, lost.world_state.lead_ttc)
        self.assertTrue(
            any("external perception unavailable" in reason for reason in lost.reasons)
        )

        recovered = planner.plan(
            SimpleNamespace(),
            vehicle_data=[],
            timestamp=1.5,
            external_perception_valid=True,
        )
        self.assertEqual(recovered.action, "KEEP")
        self.assertFalse(recovered.world_state.has_lead_vehicle)

    def test_loss_without_hazard_slows_then_brakes_and_age_is_monotonic(self) -> None:
        planner = MRMPlanner({
            "predictor": "kinematic",
            "log_enable": False,
            "safety_shield": {
                "perception_loss_min_rank": 1,
                "perception_loss_brake_after_sec": 0.5,
            },
        })
        self.assertEqual(
            planner.plan(
                SimpleNamespace(),
                vehicle_data=[],
                timestamp=2.0,
                external_perception_valid=True,
            ).action,
            "KEEP",
        )
        initial = planner.plan(
            SimpleNamespace(),
            vehicle_data=[],
            timestamp=2.1,
            external_perception_valid=False,
        )
        late = planner.plan(
            SimpleNamespace(),
            vehicle_data=[],
            timestamp=2.7,
            external_perception_valid=False,
        )
        clock_back = planner.plan(
            SimpleNamespace(),
            vehicle_data=[],
            timestamp=2.4,
            external_perception_valid=False,
        )

        self.assertEqual(initial.action, "SLOW_DOWN")
        self.assertGreaterEqual(self._rank(late.action), 2)
        self.assertGreaterEqual(
            clock_back.world_state.external_perception_age_sec,
            late.world_state.external_perception_age_sec,
        )

        stale = MRMPlanner({"predictor": "kinematic", "log_enable": False})
        stale_result = stale.plan(
            SimpleNamespace(),
            vehicle_data=[],
            timestamp=5.0,
            external_perception_valid=False,
            external_perception_age_sec=1.0,
        )
        self.assertGreaterEqual(self._rank(stale_result.action), 2)
        self.assertEqual(
            stale_result.world_state.external_perception_age_sec, 1.0
        )

    def test_loss_resets_rssm_and_bypasses_learned_prediction(self) -> None:
        class FakeRSSM:
            ready = True
            status = "ready"

            def __init__(self):
                self.reset_calls = 0
                self.observe_calls = 0
                self.predict_calls = 0

            def reset(self):
                self.reset_calls += 1

            def observe(self, *args, **kwargs):
                self.observe_calls += 1
                return True

            def predict_many(self, *args, **kwargs):
                self.predict_calls += 1
                return {}

        planner = MRMPlanner({"predictor": "kinematic", "log_enable": False})
        fake = FakeRSSM()
        planner.rssm = fake

        result = planner.plan(
            SimpleNamespace(),
            vehicle_data=[],
            timestamp=1.0,
            external_perception_valid=False,
        )
        planner.plan(
            SimpleNamespace(),
            vehicle_data=[],
            timestamp=1.1,
            external_perception_valid=False,
        )

        self.assertEqual(fake.reset_calls, 1)
        self.assertEqual(fake.observe_calls, 0)
        self.assertEqual(fake.predict_calls, 0)
        self.assertEqual(result.prediction_source, "kinematic")

    def test_reset_clears_held_hazard_and_invalid_config_is_rejected(self) -> None:
        planner = MRMPlanner({"predictor": "kinematic", "log_enable": False})
        planner.plan(
            SimpleNamespace(),
            vehicle_data=[{
                "distance": 1.0,
                "ttc": 0.5,
                "rel_speed": 2.0,
                "lane_relevance": 1.0,
                "warning_level": 2,
            }],
            timestamp=1.0,
        )
        planner.reset()
        lost = planner.plan(
            SimpleNamespace(),
            vehicle_data=[],
            timestamp=1.1,
            external_perception_valid=False,
        )
        self.assertFalse(lost.world_state.external_state_held)
        self.assertEqual(lost.action, "SLOW_DOWN")

        for value in (0, 4, 1.5, True, "1"):
            with self.subTest(rank=value), self.assertRaises(ValueError):
                MRMPlanner({
                    "safety_shield": {"perception_loss_min_rank": value}
                })
        for value in (-0.1, float("nan"), float("inf"), True):
            with self.subTest(delay=value), self.assertRaises(ValueError):
                MRMPlanner({
                    "safety_shield": {
                        "perception_loss_brake_after_sec": value
                    }
                })
        with self.assertRaisesRegex(ValueError, "must be a bool"):
            planner.plan(
                SimpleNamespace(),
                external_perception_valid=1,
            )
        for value in (-0.1, float("nan"), float("inf"), True, "bad"):
            with self.subTest(age=value), self.assertRaises(ValueError):
                planner.plan(
                    SimpleNamespace(),
                    external_perception_valid=False,
                    external_perception_age_sec=value,
                )


class AnalyticHeldActionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = WorldState(
            has_lead_vehicle=True,
            lead_distance=12.0,
            lead_ttc=1.5,
            closing_speed=8.0,
            lane_relevance=1.0,
        )
        self.predictor = RiskPredictor({"horizons_sec": [0.5, 1.0, 1.5]})
        self.actions = CandidateActions().generate(self.state)
        self.brake = next(action for action in self.actions if action.name == "BRAKE")

    def test_risk_predictor_held_brake_uses_zero_delay(self) -> None:
        onset = self.predictor.predict(self.state, self.brake)
        held = self.predictor.predict(
            self.state,
            self.brake,
            response_delay_sec=0.0,
        )

        self.assertGreater(held.min_distance, onset.min_distance)
        self.assertLessEqual(held.peak_risk, onset.peak_risk)
        self.assertLessEqual(held.terminal_risk, onset.terminal_risk)

    def test_brake_turning_point_is_included_between_requested_horizons(self) -> None:
        config = {
            "horizons_sec": [0.5, 1.0, 1.5],
            "ttc_critical": 0.1,
            "ttc_safe": 1.0,
            "dist_critical": 1.98,
            "dist_safe": 2.10,
        }
        predictor = RiskPredictor(config)
        state = WorldState(
            has_lead_vehicle=True,
            lead_distance=2.895,
            lead_ttc=2.895 / 2.1,
            closing_speed=2.1,
            lane_relevance=1.0,
        )
        brake = next(
            action
            for action in CandidateActions().generate(state)
            if action.name == "BRAKE"
        )

        direct = predictor.predict(state, brake)

        self.assertAlmostEqual(direct.min_distance, 1.95, places=3)
        self.assertGreaterEqual(direct.peak_risk, 0.99)

        planner = MRMPlanner({"predictor": "kinematic", **config})
        result = planner.plan(
            SimpleNamespace(),
            vehicle_data=[{
                "distance": 2.895,
                "ttc": 2.895 / 2.1,
                "rel_speed": 2.1,
                "lane_relevance": 1.0,
                "warning_level": 0,
            }],
            timestamp=1.0,
        )
        planned_brake = next(
            item.prediction for item in result.candidates if item.action.name == "BRAKE"
        )
        self.assertAlmostEqual(planned_brake.min_distance, 1.95, places=3)
        self.assertGreaterEqual(planned_brake.peak_risk, 0.99)

    def test_planner_only_zeroes_delay_for_matching_applied_action(self) -> None:
        planner = MRMPlanner({
            "predictor": "kinematic",
            "horizons_sec": [0.5, 1.0, 1.5],
        })
        vehicles = [{
            "distance": 12.0,
            "ttc": 1.5,
            "rel_speed": 8.0,
            "lane_relevance": 1.0,
            "warning_level": 0,
        }]
        onset = planner.plan(
            SimpleNamespace(),
            vehicle_data=vehicles,
            timestamp=1.0,
        )
        held = planner.plan(
            SimpleNamespace(),
            vehicle_data=vehicles,
            timestamp=1.1,
            applied_action={
                "name": "BRAKE",
                "target_decel": self.brake.target_decel,
                "response_delay_sec": self.brake.response_delay_sec,
            },
        )
        onset_predictions = {
            item.action.name: item.prediction for item in onset.candidates
        }
        held_predictions = {
            item.action.name: item.prediction for item in held.candidates
        }

        self.assertGreater(
            held_predictions["BRAKE"].min_distance,
            onset_predictions["BRAKE"].min_distance,
        )
        self.assertLessEqual(
            held_predictions["BRAKE"].peak_risk,
            onset_predictions["BRAKE"].peak_risk,
        )
        for name in ("KEEP", "SLOW_DOWN", "EMERGENCY_BRAKE"):
            with self.subTest(candidate=name):
                self.assertEqual(held_predictions[name], onset_predictions[name])

    def test_planner_accepts_remaining_delay_longer_than_one_model_step(self) -> None:
        planner = MRMPlanner({"predictor": "kinematic"})
        state = WorldState(
            has_lead_vehicle=True,
            lead_distance=21.56,
            lead_ttc=21.56 / 11.96,
            closing_speed=11.96,
            lane_relevance=1.0,
        )
        brake = next(
            action
            for action in CandidateActions().generate(state)
            if action.name == "BRAKE"
        )
        receipt = {
            "name": brake.name,
            "target_decel": brake.target_decel,
            "remaining_delay_sec": 0.30,
        }

        self.assertAlmostEqual(
            planner._applied_action_delay_override(brake, receipt), 0.30
        )
        expected = planner.predictor.predict(
            state, brake, response_delay_sec=0.30
        )
        actual = planner.plan(
            SimpleNamespace(),
            vehicle_data=[{
                "distance": state.lead_distance,
                "ttc": state.lead_ttc,
                "rel_speed": state.closing_speed,
                "lane_relevance": 1.0,
                "warning_level": 0,
            }],
            timestamp=1.0,
            applied_action=receipt,
        )
        brake_prediction = next(
            item.prediction for item in actual.candidates if item.action.name == "BRAKE"
        )
        self.assertEqual(brake_prediction, expected)

    def test_planner_preserves_confirmed_action_remaining_delay(self) -> None:
        planner = MRMPlanner({
            "predictor": "kinematic",
            "horizons_sec": [0.5, 1.0, 1.5],
        })
        vehicles = [{
            "distance": 12.0,
            "ttc": 1.5,
            "rel_speed": 8.0,
            "lane_relevance": 1.0,
            "warning_level": 0,
        }]
        result = planner.plan(
            SimpleNamespace(),
            vehicle_data=vehicles,
            timestamp=1.0,
            applied_action={
                "name": "BRAKE",
                "target_decel": self.brake.target_decel,
                "remaining_delay_sec": 0.10,
            },
        )
        actual = next(
            item.prediction for item in result.candidates if item.action.name == "BRAKE"
        )
        expected = self.predictor.predict(
            self.state,
            self.brake,
            response_delay_sec=0.10,
        )
        onset = self.predictor.predict(self.state, self.brake)
        held = self.predictor.predict(self.state, self.brake, response_delay_sec=0.0)

        self.assertEqual(actual, expected)
        self.assertGreater(actual.min_distance, onset.min_distance)
        self.assertLess(actual.min_distance, held.min_distance)

    def test_action_changed_receipt_is_held_for_next_transition(self) -> None:
        planner = MRMPlanner({
            "predictor": "kinematic",
            "horizons_sec": [0.5, 1.0, 1.5],
        })
        result = planner.plan(
            SimpleNamespace(),
            vehicle_data=[{
                "distance": 12.0,
                "ttc": 1.5,
                "rel_speed": 8.0,
                "lane_relevance": 1.0,
                "warning_level": 0,
            }],
            timestamp=1.0,
            applied_action={
                "name": "BRAKE",
                "target_decel": self.brake.target_decel,
                "response_delay_sec": self.brake.response_delay_sec,
                "action_changed": True,
            },
        )
        actual = next(
            item.prediction for item in result.candidates if item.action.name == "BRAKE"
        )
        held = self.predictor.predict(
            self.state, self.brake, response_delay_sec=0.0
        )

        self.assertEqual(actual, held)


class RiskCostConfigurationTests(unittest.TestCase):
    def test_rejects_negative_risk_weight(self) -> None:
        weight_names = (
            "pred_weight",
            "terminal_weight",
            "fusion_weight",
            "driver_weight",
        )
        for name in weight_names:
            with (
                self.subTest(weight=name),
                self.assertRaisesRegex(ValueError, "weights must be non-negative"),
            ):
                RiskCost({name: -0.01})

    def test_rejects_driver_thresholds_outside_range_or_out_of_order(self) -> None:
        cases = (
            {"driver_slowdown_threshold": -0.01},
            {"driver_high_threshold": 1.01},
            {"driver_slowdown_threshold": 0.80, "driver_high_threshold": 0.70},
        )
        for config in cases:
            with (
                self.subTest(config=config),
                self.assertRaisesRegex(ValueError, "0 <= slowdown <= high <= 1"),
            ):
                RiskCost(config)


class SafetyShieldConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predictor = RiskPredictor({})

    def test_rejects_negative_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            SafetyShield(
                {"safety_shield": {"emergency_ttc": -0.01}},
                self.predictor,
            )

    def test_rejects_out_of_order_ttc_or_distance_thresholds(self) -> None:
        cases = (
            {"emergency_ttc": 1.3, "brake_ttc": 1.2, "caution_ttc": 3.0},
            {
                "emergency_distance": 3.1,
                "brake_distance": 3.0,
                "caution_distance": 10.0,
            },
        )
        for shield_config in cases:
            with (
                self.subTest(shield_config=shield_config),
                self.assertRaisesRegex(ValueError, "must satisfy"),
            ):
                SafetyShield({"safety_shield": shield_config}, self.predictor)

    def test_rejects_driver_threshold_outside_unit_interval(self) -> None:
        for threshold in (-0.01, 1.01):
            with (
                self.subTest(threshold=threshold),
                self.assertRaisesRegex(ValueError, "must be in \\[0, 1\\]"),
            ):
                SafetyShield(
                    {"safety_shield": {"driver_slowdown_threshold": threshold}},
                    self.predictor,
                )


class RSSMActionCatalogFallbackTests(unittest.TestCase):
    @staticmethod
    def _catalog() -> list[dict[str, float | str]]:
        actions = CandidateActions().generate(WorldState())
        return [
            {
                "name": action.name,
                "target_decel": action.target_decel,
                "response_delay_sec": action.response_delay_sec,
            }
            for action in actions
        ]

    @staticmethod
    def _save_checkpoint(path: Path, metrics: dict[str, object]) -> None:
        model = TinyRSSM(
            RSSMConfig(
                embed_dim=8,
                deter_dim=8,
                stoch_dim=2,
                classes=2,
                hidden_dim=8,
                samples=2,
            )
        )
        model.save_checkpoint(path, metrics=metrics)

    @staticmethod
    def _action_contract() -> dict[str, object]:
        return {
            "version": ACTION_CONTRACT_VERSION,
            "alignment": "previous_observation_applied_action_current_observation",
            "unknown_action_supported": True,
        }

    @classmethod
    def _valid_metrics(cls) -> dict[str, object]:
        catalog = cls._catalog()
        return {
            "steps": 1000,
            "action_catalog": catalog,
            "action_contract": cls._action_contract(),
            "unknown_action_training": {"unknown_fraction": 0.25},
            "known_executed_action_counts": {
                str(item["name"]): 1 for item in catalog
            },
            "fusion_feature_contract": build_fusion_feature_contract(),
        }

    @staticmethod
    def _planner(checkpoint: Path) -> MRMPlanner:
        return MRMPlanner({
            "predictor": "hybrid",
            "rssm": {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256(checkpoint),
                "device": "cpu",
            },
        })

    def _assert_kinematic_fallback(self, checkpoint: Path) -> None:
        planner = self._planner(checkpoint)
        self.assertIsNone(planner.rssm)
        self.assertTrue(planner.model_status.startswith("fallback"))

        result = planner.plan(SimpleNamespace(), timestamp=1.0)
        self.assertEqual(result.prediction_source, "kinematic")
        self.assertFalse(any("RSSM" in reason for reason in result.reasons))

    def test_missing_action_catalog_falls_back_without_using_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "missing_catalog.pt"
            metrics = self._valid_metrics()
            metrics.pop("action_catalog")
            self._save_checkpoint(checkpoint, metrics=metrics)
            self._assert_kinematic_fallback(checkpoint)

    def test_mismatched_action_catalog_falls_back_without_using_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "mismatched_catalog.pt"
            metrics = self._valid_metrics()
            catalog = metrics["action_catalog"]
            self.assertIsInstance(catalog, list)
            catalog[2]["target_decel"] = 4.0
            self._save_checkpoint(checkpoint, metrics=metrics)
            self._assert_kinematic_fallback(checkpoint)

    def test_matching_catalog_without_action_contract_metadata_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            cases = (
                (
                    "missing_action_contract.pt",
                    {
                        "action_catalog": self._catalog(),
                        "unknown_action_training": {"unknown_fraction": 0.25},
                    },
                ),
                (
                    "missing_unknown_training.pt",
                    {
                        "action_catalog": self._catalog(),
                        "action_contract": self._action_contract(),
                    },
                ),
                (
                    "missing_known_action_coverage.pt",
                    {
                        "action_catalog": self._catalog(),
                        "action_contract": self._action_contract(),
                        "unknown_action_training": {"unknown_fraction": 0.25},
                    },
                ),
            )
            for filename, metrics in cases:
                with self.subTest(filename=filename):
                    checkpoint = base / filename
                    self._save_checkpoint(checkpoint, metrics=metrics)
                    self._assert_kinematic_fallback(checkpoint)

    def test_missing_or_mismatched_fusion_contract_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            missing = self._valid_metrics()
            missing.pop("fusion_feature_contract")
            missing_path = base / "missing_fusion_contract.pt"
            self._save_checkpoint(missing_path, metrics=missing)
            self._assert_kinematic_fallback(missing_path)

            mismatched = self._valid_metrics()
            mismatched["fusion_feature_contract"] = build_fusion_feature_contract(
                {"w_ext": 0.50}
            )
            mismatch_path = base / "mismatched_fusion_contract.pt"
            self._save_checkpoint(mismatch_path, metrics=mismatched)
            self._assert_kinematic_fallback(mismatch_path)

    def test_insufficient_training_steps_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "undertrained.pt"
            metrics = self._valid_metrics()
            metrics["steps"] = 999
            self._save_checkpoint(checkpoint, metrics=metrics)

            self._assert_kinematic_fallback(checkpoint)

    def test_invalid_minimum_training_steps_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "invalid-minimum.pt"
            self._save_checkpoint(checkpoint, metrics=self._valid_metrics())

            for value in (0, -1, 999, 1.5, True, "1000"):
                with self.subTest(value=value):
                    planner = MRMPlanner({
                        "predictor": "hybrid",
                        "rssm": {
                            "checkpoint": str(checkpoint),
                            "checkpoint_sha256": checkpoint_sha256(checkpoint),
                            "device": "cpu",
                            "min_training_steps": value,
                        },
                    })
                    self.assertIsNone(planner.rssm)
                    self.assertTrue(planner.model_status.startswith("fallback"))

    def test_missing_or_mismatched_checkpoint_digest_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "digest.pt"
            self._save_checkpoint(checkpoint, metrics=self._valid_metrics())

            missing = MRMPlanner({
                "predictor": "hybrid",
                "rssm": {"checkpoint": str(checkpoint), "device": "cpu"},
            })
            wrong = MRMPlanner({
                "predictor": "hybrid",
                "rssm": {
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": "0" * 64,
                    "device": "cpu",
                },
            })

            for planner in (missing, wrong):
                self.assertIsNone(planner.rssm)
                self.assertTrue(planner.model_status.startswith("fallback"))

    def test_complete_matching_action_contract_is_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "matching_contract.pt"
            self._save_checkpoint(checkpoint, metrics=self._valid_metrics())

            planner = self._planner(checkpoint)

            self.assertIsNotNone(planner.rssm)
            self.assertTrue(planner.rssm.ready)
            self.assertEqual(planner.model_status, "ready")

    def test_checkpoint_capacity_mismatch_falls_back_during_initialization(self) -> None:
        cases = (
            ("max_decel", {"max_decel": 6.0}),
            ("max_delay", {"max_delay": 0.15}),
            ("dt_sec", {"dt_sec": 0.10}),
        )
        base_config = {
            "embed_dim": 8,
            "deter_dim": 8,
            "stoch_dim": 2,
            "classes": 2,
            "hidden_dim": 8,
            "samples": 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, override in cases:
                with self.subTest(capacity=name):
                    checkpoint = Path(directory) / f"invalid_{name}.pt"
                    values = dict(base_config)
                    values.update(override)
                    model = TinyRSSM(RSSMConfig(**values))
                    model.save_checkpoint(checkpoint, metrics=self._valid_metrics())

                    planner = self._planner(checkpoint)

                    self.assertIsNone(planner.rssm)
                    self.assertTrue(planner.model_status.startswith("fallback"))


if __name__ == "__main__":
    unittest.main()
