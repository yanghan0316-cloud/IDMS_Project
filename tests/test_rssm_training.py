"""Fast CPU regression tests for RSSM synthetic training dynamics."""

from __future__ import annotations

import unittest

import torch

from src.core.rssm_world_model import (
    ActionCodec,
    TinyRSSM,
    WorldStateCodec,
    build_fusion_feature_contract,
)
from train_rssm import (
    _encode_observation,
    _integrate_nonnegative_speed,
    load_feature_contract,
    synthetic_batch,
    training_loss,
)


class NonnegativeSpeedIntegrationTests(unittest.TestCase):
    def test_constant_deceleration_and_stopping_without_reverse(self) -> None:
        speed = torch.tensor([10.0, 2.0, 0.0])
        acceleration = torch.tensor([-2.0, -4.0, -3.0])
        duration = torch.tensor([3.0, 2.0, 1.0])

        next_speed, displacement = _integrate_nonnegative_speed(
            speed,
            acceleration,
            duration,
        )

        torch.testing.assert_close(next_speed, torch.tensor([4.0, 0.0, 0.0]))
        torch.testing.assert_close(displacement, torch.tensor([21.0, 0.5, 0.0]))
        self.assertTrue((next_speed >= 0.0).all().item())
        self.assertTrue((displacement >= 0.0).all().item())

    def test_onset_delay_displacement_uses_squared_active_time(self) -> None:
        speed = torch.tensor([10.0])
        deceleration = 4.0
        dt = 0.25
        response_delay = 0.10
        active_time = dt - response_delay

        next_speed, active_displacement = _integrate_nonnegative_speed(
            speed,
            torch.tensor([-deceleration]),
            active_time,
        )
        displacement = active_displacement + speed * response_delay

        expected = speed * dt - 0.5 * deceleration * active_time**2
        incorrect_linear_term = speed * dt - 0.5 * deceleration * active_time * dt
        torch.testing.assert_close(next_speed, torch.tensor([9.4]))
        torch.testing.assert_close(displacement, expected)
        self.assertFalse(
            torch.isclose(displacement, incorrect_linear_term, atol=1e-6).all().item()
        )


class FusionFeatureContractTests(unittest.TestCase):
    def test_contract_covers_driver_scoring_configuration(self) -> None:
        default = build_fusion_feature_contract()
        changed = build_fusion_feature_contract({
            "corroboration_boost": 3.0,
            "w_perclos": 0.40,
        })

        self.assertNotEqual(default, changed)
        self.assertEqual(changed["driver"]["corroboration_boost"], 3.0)
        self.assertEqual(changed["driver"]["w_perclos"], 0.40)

        upstream_changed = build_fusion_feature_contract(
            internal_config={"fps": 30.0}
        )
        self.assertNotEqual(default, upstream_changed)
        self.assertEqual(upstream_changed["upstream_internal"]["fps"], 30.0)
        self.assertEqual(load_feature_contract("none"), default)

    def test_synthetic_features_match_online_default_fusion_equations(self) -> None:
        codec = WorldStateCodec()
        encoded, _, fused = _encode_observation(
            codec=codec,
            has_lead_vehicle=torch.tensor([True]),
            distance_m=torch.tensor([10.0]),
            closing_speed_mps=torch.tensor([2.0]),
            lane_relevance=torch.tensor([0.8]),
            fatigue_score=torch.tensor([0.4]),
            attention_score=torch.tensor([0.5]),
        )
        ttc_score = 1.0 - (5.0 - 1.5) / (6.0 - 1.5)
        distance_score = 1.0 - (10.0 - 3.0) / (30.0 - 3.0)
        external = max(ttc_score, distance_score) * 0.8
        internal = 0.55 * 0.4 + 0.45 * 0.5 + 0.20 * 0.4 * 0.5
        cross = external * internal
        expected_fused = 0.35 * external + 0.35 * internal + 0.30 * cross

        self.assertAlmostEqual(float(encoded[0, 6]), external, places=6)
        self.assertAlmostEqual(float(encoded[0, 7]), internal, places=6)
        self.assertAlmostEqual(float(encoded[0, 8]), cross, places=6)
        self.assertAlmostEqual(float(encoded[0, 9]), expected_fused, places=6)
        self.assertAlmostEqual(float(fused[0]), expected_fused, places=6)

        no_risk, _, next_fused = _encode_observation(
            codec=codec,
            has_lead_vehicle=torch.tensor([False]),
            distance_m=torch.tensor([99.0]),
            closing_speed_mps=torch.tensor([0.0]),
            lane_relevance=torch.tensor([1.0]),
            fatigue_score=torch.tensor([0.0]),
            attention_score=torch.tensor([0.0]),
            previous_fused_score=fused,
        )
        self.assertAlmostEqual(float(no_risk[0, 9]), 0.60 * expected_fused, places=6)
        self.assertAlmostEqual(float(next_fused[0]), 0.60 * expected_fused, places=6)


class SyntheticBatchTests(unittest.TestCase):
    def test_transition_shapes_finite_values_and_unknown_action_mask(self) -> None:
        batch_size = 32
        sequence_length = 6
        generator = torch.Generator(device="cpu").manual_seed(123)
        world_codec = WorldStateCodec()
        action_codec = ActionCodec()

        batch = synthetic_batch(
            batch_size=batch_size,
            sequence_length=sequence_length,
            generator=generator,
            device=torch.device("cpu"),
            world_codec=world_codec,
            action_codec=action_codec,
        )

        self.assertEqual(
            batch["observations"].shape,
            (batch_size, sequence_length + 1, len(world_codec.OBS_FIELDS)),
        )
        self.assertEqual(
            batch["executed_actions"].shape,
            (batch_size, sequence_length, 3),
        )
        self.assertEqual(
            batch["executed_action_ids"].shape,
            (batch_size, sequence_length),
        )
        self.assertEqual(
            batch["risk_targets"].shape,
            (batch_size, sequence_length + 1),
        )
        self.assertEqual(
            batch["continues"].shape,
            (batch_size, sequence_length + 1),
        )
        self.assertEqual(
            batch["valid_mask"].shape,
            (batch_size, sequence_length + 1),
        )

        for name, tensor in batch.items():
            self.assertEqual(tensor.device.type, "cpu", name)
            self.assertTrue(torch.isfinite(tensor).all().item(), name)

        actions = batch["executed_actions"]
        valid_mask = actions[..., -1]
        self.assertTrue(((valid_mask == 0.0) | (valid_mask == 1.0)).all().item())
        self.assertTrue((valid_mask == 0.0).any().item())
        self.assertTrue((valid_mask == 1.0).any().item())
        self.assertTrue(
            torch.equal(
                actions[valid_mask == 0.0],
                torch.zeros_like(actions[valid_mask == 0.0]),
            )
        )

    def test_collision_is_absorbing_and_masks_first_collision_only(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(2026)
        batch = synthetic_batch(
            batch_size=192,
            sequence_length=40,
            generator=generator,
            device=torch.device("cpu"),
            world_codec=WorldStateCodec(),
            action_codec=ActionCodec(),
        )
        observations = batch["observations"]
        has_lead = observations[..., 0] >= 0.5
        zero_distance = observations[..., 1] == 0.0
        collision_states = has_lead & zero_distance
        collision_rows = torch.where(collision_states.any(dim=1))[0]
        self.assertGreater(collision_rows.numel(), 0)

        continues = batch["continues"]
        valid_mask = batch["valid_mask"]
        for row_tensor in collision_rows:
            row = int(row_tensor)
            first_collision = int(torch.where(collision_states[row])[0][0])
            self.assertGreater(first_collision, 0)
            self.assertTrue(has_lead[row, first_collision:].all().item())
            self.assertTrue(zero_distance[row, first_collision:].all().item())
            self.assertTrue((continues[row, :first_collision] == 1.0).all().item())
            self.assertTrue((continues[row, first_collision:] == 0.0).all().item())
            self.assertTrue(
                (valid_mask[row, : first_collision + 1] == 1.0).all().item()
            )
            if first_collision + 1 < valid_mask.shape[1]:
                self.assertTrue(
                    (valid_mask[row, first_collision + 1 :] == 0.0).all().item()
                )

        no_lead_rows = torch.where((~has_lead).all(dim=1))[0]
        self.assertGreater(no_lead_rows.numel(), 0)
        self.assertTrue((continues[no_lead_rows] == 1.0).all().item())
        self.assertTrue((valid_mask[no_lead_rows] == 1.0).all().item())

        masked_rows = torch.where((valid_mask == 0.0).any(dim=1))[0][:4]
        self.assertGreater(masked_rows.numel(), 0)
        original = {
            name: tensor[masked_rows].clone()
            for name, tensor in batch.items()
        }
        altered = {name: tensor.clone() for name, tensor in original.items()}
        masked_tail = altered["valid_mask"] == 0.0
        altered["risk_targets"][masked_tail] = (
            1.0 - altered["risk_targets"][masked_tail]
        )
        altered["continues"][masked_tail] = 1.0 - altered["continues"][masked_tail]

        model = TinyRSSM(
            {
                "embed_dim": 16,
                "deter_dim": 16,
                "stoch_dim": 2,
                "classes": 4,
                "hidden_dim": 16,
                "samples": 2,
            }
        ).cpu()
        torch.manual_seed(77)
        original_loss, _ = training_loss(model, original)
        torch.manual_seed(77)
        altered_loss, _ = training_loss(model, altered)
        torch.testing.assert_close(original_loss, altered_loss)


if __name__ == "__main__":
    unittest.main()
