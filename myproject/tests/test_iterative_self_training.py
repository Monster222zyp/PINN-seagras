from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import train_iterative_self_training as iterative
import train_latent_physics_pinn as base


def make_experimental_data() -> base.LoadedData:
    rows = []
    config_index = []
    for config_id, elastic in enumerate([4.8e5, 3.55e6]):
        for velocity_index, u in enumerate([0.05, 0.10, 0.15]):
            rows.append(
                [
                    u,
                    0.0,
                    0.0,
                    elastic,
                    0.01,
                    0.0005,
                    30.0 + 10.0 * config_id,
                    45.0,
                    60.0,
                    0.025,
                    0.23,
                    0.08,
                    0.20,
                    0.0275,
                    5.0,
                    2.0,
                    1.2,
                ]
            )
            config_index.append(config_id)
    raw_x = iterative.recompute_dimensionless(np.asarray(rows, dtype=np.float32))
    y = np.zeros((len(raw_x), len(base.TARGET_NAMES_27)), dtype=np.float32)
    y[:, 0] = 20.0 * raw_x[:, 0] ** 2
    return base.LoadedData(
        raw_x=raw_x,
        y=y,
        config_index=np.asarray(config_index, dtype=np.int64),
        velocity_index=np.tile(np.arange(3, dtype=np.int64), 2),
        source_id=np.zeros(len(raw_x), dtype=np.int64),
        sample_weight=np.ones(len(raw_x), dtype=np.float32),
        aux_weight=np.ones(len(raw_x), dtype=np.float32),
        feature_names=list(base.FEATURE_NAMES_17),
        target_names=list(base.TARGET_NAMES_27),
        config_names=["config_0", "config_1"],
    )


def make_predictions(raw_x: np.ndarray) -> dict[str, np.ndarray]:
    n = len(raw_x)
    physics = (10.0 * raw_x[:, 0] ** 2 + 0.1).reshape(-1, 1).astype(np.float32)
    residual = np.zeros((n, 1), dtype=np.float32)
    return {
        "force": physics + residual,
        "F_physics": physics,
        "F_residual": residual,
        "Cd_stem_eff": raw_x[:, 16:17].copy(),
        "Cd_leaf_eff": raw_x[:, 15:16].copy(),
        "shielding_coef": np.full((n, 1), 0.8, dtype=np.float32),
        "reconfiguration_factor": np.full((n, 1), 0.9, dtype=np.float32),
        "reconfiguration_gain": np.full((n, 1), 0.9, dtype=np.float32),
    }


class IterativeSelfTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = make_experimental_data()
        features, _ = base.build_features(self.data.raw_x)
        self.scaler = base.Standardizer.fit(features)

    def test_candidate_generation_preserves_schema_and_dimensionless_numbers(self) -> None:
        candidates, config_ids = iterative.generate_candidate_pool(
            self.data,
            np.arange(len(self.data.raw_x), dtype=np.int64),
            n_candidates=20,
            seed=13,
            u_min=0.055,
            u_max=0.145,
        )
        self.assertEqual(candidates.shape, (20, 17))
        expected_re = base.RHO_DEFAULT * candidates[:, 0] * candidates[:, 9] / base.MU_WATER
        expected_ca = (
            6.0
            * base.RHO_DEFAULT
            * candidates[:, 15]
            * candidates[:, 0] ** 2
            * candidates[:, 11] ** 3
            / (candidates[:, 3] * candidates[:, 5] ** 3)
        )
        np.testing.assert_allclose(candidates[:, 1], expected_re, rtol=1e-6)
        np.testing.assert_allclose(candidates[:, 2], expected_ca, rtol=1e-6)
        counts = np.unique(config_ids, return_counts=True)[1]
        self.assertLessEqual(int(np.max(counts) - np.min(counts)), 1)

    def test_candidate_generation_uses_each_configuration_velocity_domain(self) -> None:
        data = make_experimental_data()
        data.raw_x[:3, 0] = np.array([0.05, 0.08, 0.11], dtype=np.float32)
        data.raw_x[3:, 0] = np.array([0.20, 0.25, 0.30], dtype=np.float32)
        data.raw_x = iterative.recompute_dimensionless(data.raw_x)
        candidates, config_ids = iterative.generate_candidate_pool(
            data,
            np.arange(len(data.raw_x), dtype=np.int64),
            n_candidates=20,
            seed=9,
        )
        self.assertTrue(np.all((candidates[config_ids == 0, 0] >= 0.05) & (candidates[config_ids == 0, 0] <= 0.11)))
        self.assertTrue(np.all((candidates[config_ids == 1, 0] >= 0.20) & (candidates[config_ids == 1, 0] <= 0.30)))

        seen = {
            int(iterative.generate_candidate_pool(data, np.arange(6), 1, seed)[1][0])
            for seed in range(12)
        }
        self.assertEqual(seen, {0, 1})

    def test_filter_rejects_duplicates_and_large_residual(self) -> None:
        raw_x = self.data.raw_x[:3].copy()
        raw_x[:, 0] = np.array([0.071, 0.082, 0.093], dtype=np.float32)
        raw_x = iterative.recompute_dimensionless(raw_x)
        predictions = make_predictions(raw_x)
        predictions["F_residual"][1, 0] = 0.3 * predictions["F_physics"][1, 0]
        predictions["force"][1, 0] += predictions["F_residual"][1, 0]
        evaluation = iterative.filter_candidate_pool(
            raw_x,
            np.zeros(3, dtype=np.int64),
            predictions,
            self.scaler,
            reference_raw_x=raw_x[:1],
            max_residual_ratio=0.2,
            max_abs_feature_z=100.0,
            max_cd_ratio=3.0,
        )
        self.assertIn("duplicate", evaluation.rejection_reason[0])
        self.assertIn("residual_ratio", evaluation.rejection_reason[1])
        self.assertTrue(evaluation.filter_pass[2])
        selected = iterative.select_balanced_candidates(evaluation, target_count=1)
        np.testing.assert_array_equal(selected, np.array([2]))

    def test_selection_remainder_uses_confidence_and_clears_previous_state(self) -> None:
        raw_x = self.data.raw_x[:2].copy()
        evaluation = iterative.filter_candidate_pool(
            raw_x,
            np.array([0, 1], dtype=np.int64),
            make_predictions(raw_x),
            self.scaler,
            reference_raw_x=np.empty((0, 17), dtype=np.float32),
            max_residual_ratio=0.2,
            max_abs_feature_z=100.0,
            max_cd_ratio=3.0,
        )
        evaluation.confidence_score[:] = np.array([0.2, 0.9])
        iterative.select_balanced_candidates(evaluation, target_count=2)
        selected = iterative.select_balanced_candidates(evaluation, target_count=1)
        np.testing.assert_array_equal(selected, np.array([1]))
        np.testing.assert_array_equal(evaluation.selected, np.array([False, True]))
        self.assertEqual(evaluation.rejection_reason[0], "valid_not_selected")
        self.assertEqual(evaluation.rejection_reason[1], "accepted")

        evaluation.filter_pass[:] = False
        empty = iterative.select_balanced_candidates(evaluation, target_count=1)
        self.assertEqual(len(empty), 0)
        self.assertFalse(np.any(evaluation.selected))

    def test_pseudo_rows_are_force_only_and_have_explicit_provenance(self) -> None:
        raw_x = self.data.raw_x[:2].copy()
        predictions = make_predictions(raw_x)
        evaluation = iterative.filter_candidate_pool(
            raw_x,
            np.array([0, 1], dtype=np.int64),
            predictions,
            self.scaler,
            reference_raw_x=np.empty((0, 17), dtype=np.float32),
            max_residual_ratio=0.2,
            max_abs_feature_z=100.0,
            max_cd_ratio=3.0,
        )
        selected = iterative.select_balanced_candidates(evaluation, target_count=2)
        pseudo = iterative.build_pseudo_data(evaluation, selected, cycle_id=3, pseudo_label_weight=0.2)
        np.testing.assert_allclose(pseudo.y[:, 0], predictions["force"].reshape(-1))
        np.testing.assert_array_equal(pseudo.y[:, 1:], np.zeros_like(pseudo.y[:, 1:]))
        np.testing.assert_array_equal(pseudo.source_id, np.full(2, iterative.PSEUDO_SOURCE_ID))
        np.testing.assert_allclose(pseudo.sample_weight, 0.2)
        np.testing.assert_allclose(pseudo.aux_weight, 0.0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "pseudo.h5"
            iterative.write_pseudo_h5(path, pseudo, cycle_id=3)
            loaded = base.load_dataset(path)
            self.assertEqual(loaded.raw_x.shape, (2, 17))
            np.testing.assert_array_equal(loaded.source_id, np.full(2, iterative.PSEUDO_SOURCE_ID))
            np.testing.assert_array_equal(loaded.config_index, pseudo.config_index)
            np.testing.assert_array_equal(loaded.velocity_index, pseudo.velocity_index)
            self.assertEqual(loaded.config_names, pseudo.config_names)
            with h5py.File(path, "r") as handle:
                np.testing.assert_array_equal(handle["pinn_data/cycle_id"][:], np.full((2, 1), 3))

    def test_pseudo_force_weight_is_not_canceled_within_a_batch(self) -> None:
        pred = torch.ones((3, 1), dtype=torch.float32)
        target = torch.zeros_like(pred)
        scale = torch.ones((), dtype=torch.float32)
        pseudo_weight = torch.full_like(pred, 0.2)
        full_weight = torch.ones_like(pred)

        pseudo_loss = base.normalized_mse(pred, target, scale, pseudo_weight)
        full_loss = base.normalized_mse(pred, target, scale, full_weight)
        torch.testing.assert_close(pseudo_loss, torch.tensor(0.2))
        torch.testing.assert_close(full_loss, torch.tensor(1.0))

    def test_validation_degradation_is_capped_against_global_best(self) -> None:
        threshold = iterative.validation_acceptance_threshold(1.0, 0.02)
        self.assertAlmostEqual(threshold, 1.02)
        self.assertGreater(1.02 * 1.02, threshold)

    def test_checkpoint_promotion_and_rejection_semantics(self) -> None:
        best_payload = {
            "model_state": {"weight": torch.tensor([1.0])},
            "meta": {
                "normalization": {"feature_mean": [0.0], "feature_std": [1.0]},
                "feature_names": ["U"],
                "target_names": ["force"],
            },
        }
        current_payload = {
            "model_state": {"weight": torch.tensor([2.0])},
            "meta": best_payload["meta"],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            best_path = run_dir / "best_source.pt"
            current_path = run_dir / "current_source.pt"
            torch.save(best_payload, best_path)
            torch.save(current_payload, current_path)

            promoted = iterative.promote_run_checkpoints(run_dir, best_path, current_path)
            root_payload = torch.load(promoted["model"], map_location="cpu", weights_only=True)
            final_payload = torch.load(promoted["final_model"], map_location="cpu", weights_only=True)
            accepted_payload = torch.load(
                promoted["last_accepted_model"], map_location="cpu", weights_only=True
            )
            self.assertEqual(root_payload["meta"]["normalization"]["feature_mean"], [0.0])
            torch.testing.assert_close(root_payload["model_state"]["weight"], torch.tensor([1.0]))
            torch.testing.assert_close(final_payload["model_state"]["weight"], torch.tensor([1.0]))
            torch.testing.assert_close(accepted_payload["model_state"]["weight"], torch.tensor([2.0]))

            attempt_dir = run_dir / "cycle_01_posttrain"
            attempt_dir.mkdir()
            attempted_path = attempt_dir / "model.pt"
            torch.save(current_payload, attempted_path)
            rejected_path = iterative.mark_rejected_checkpoint(attempted_path)
            self.assertEqual(rejected_path.name, "rejected_model.pt")
            self.assertFalse(attempted_path.exists())
            self.assertTrue(rejected_path.exists())

    def test_fixed_validation_indices_never_enter_posttraining(self) -> None:
        data = make_experimental_data()
        data.source_id[:] = np.array([0, 2, 0, 0, 2, 0], dtype=np.int64)
        experimental_train = np.array([0, 2, 3], dtype=np.int64)
        validation = np.array([5], dtype=np.int64)
        train = iterative.combined_training_indices(data, experimental_train)
        np.testing.assert_array_equal(train, np.array([0, 2, 3, 1, 4]))
        self.assertTrue(set(train).isdisjoint(set(validation)))

    def test_small_split_keeps_one_experimental_training_row(self) -> None:
        data = make_experimental_data()
        keep = np.array([0, 1])
        two_rows = base.LoadedData(
            raw_x=data.raw_x[keep],
            y=data.y[keep],
            config_index=data.config_index[keep],
            velocity_index=data.velocity_index[keep],
            source_id=data.source_id[keep],
            sample_weight=data.sample_weight[keep],
            aux_weight=data.aux_weight[keep],
            feature_names=data.feature_names,
            target_names=data.target_names,
            config_names=data.config_names,
        )
        train, validation = base.split_experimental_random(two_rows, val_ratio=0.99, seed=1)
        self.assertEqual(len(train), 1)
        self.assertEqual(len(validation), 1)


if __name__ == "__main__":
    unittest.main()
