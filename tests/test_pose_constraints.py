"""姿态约束精修单元测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.pose_constraints import (
    H36M_TOPOLOGY,
    apply_length_symmetry,
    bone_key,
    clamp_hinge_angle,
    compare_pose_sequences,
    count_angle_violations,
    default_refinement_config,
    detect_low_score,
    detect_speed_outliers,
    fill_targets_with_standard_ratios,
    filter_sequence_one_euro,
    interpolate_gaps,
    load_refinement_config,
    median_bone_lengths,
    project_bone_lengths,
    refine_2d_keypoints,
    refine_3d_poses,
    sequence_bone_lengths,
)


def _standing_pose(scale: float = 100.0) -> np.ndarray:
    pose = np.zeros((17, 3), dtype=np.float64)
    pose[1] = [scale * 0.12, 0, 0]          # rHip
    pose[2] = [scale * 0.12, scale * 0.45, 0]  # rKnee
    pose[3] = [scale * 0.12, scale * 0.90, 0]  # rAnkle
    pose[4] = [-scale * 0.12, 0, 0]
    pose[5] = [-scale * 0.12, scale * 0.45, 0]
    pose[6] = [-scale * 0.12, scale * 0.90, 0]
    pose[7] = [0, -scale * 0.20, 0]
    pose[8] = [0, -scale * 0.40, 0]
    pose[9] = [0, -scale * 0.52, 0]
    pose[10] = [0, -scale * 0.68, 0]
    pose[11] = [-scale * 0.22, -scale * 0.40, 0]
    pose[12] = [-scale * 0.22, -scale * 0.18, 0]
    pose[13] = [-scale * 0.22, 0.02 * scale, 0]
    pose[14] = [scale * 0.22, -scale * 0.40, 0]
    pose[15] = [scale * 0.22, -scale * 0.18, 0]
    pose[16] = [scale * 0.22, 0.02 * scale, 0]
    return pose


class TestConfig(unittest.TestCase):
    def test_default_and_file_merge(self):
        cfg = load_refinement_config(str(ROOT / "configs" / "pose_refinement_default.json"))
        self.assertEqual(cfg["2d"]["max_gap"], 8)
        self.assertTrue(cfg["3d"]["angle"]["enabled"])
        self.assertEqual(default_refinement_config()["fps"], 30.0)


class Test2DRepair(unittest.TestCase):
    def test_low_score_and_gap_interpolation(self):
        t, j = 12, 17
        coords = np.zeros((t, j, 2), dtype=np.float64)
        scores = np.ones((t, j), dtype=np.float64)
        for i in range(t):
            coords[i, 3] = [float(i), 0.0]
        scores[4:8, 3] = 0.05
        valid = ~detect_low_score(scores, 0.25)
        self.assertFalse(valid[5, 3])
        new_coords, new_scores, new_valid, repaired = interpolate_gaps(
            coords, scores, valid, max_gap=8, repaired_score=0.4
        )
        self.assertTrue(np.all(repaired[4:8, 3]))
        np.testing.assert_allclose(new_coords[6, 3], [6.0, 0.0], atol=1e-6)
        self.assertGreaterEqual(new_scores[6, 3], 0.39)
        self.assertTrue(new_valid[6, 3])

    def test_long_gap_not_filled(self):
        coords = np.zeros((10, 17, 2))
        scores = np.ones((10, 17))
        scores[1:9, 2] = 0.0
        valid = ~detect_low_score(scores, 0.25)
        _, _, new_valid, repaired = interpolate_gaps(coords, scores, valid, max_gap=3, repaired_score=0.4)
        self.assertFalse(np.any(repaired[:, 2]))
        self.assertFalse(np.any(new_valid[1:9, 2]))

    def test_speed_outliers(self):
        coords = np.zeros((20, 17, 2))
        valid = np.ones((20, 17), dtype=bool)
        for i in range(20):
            coords[i, 6] = [i * 1.0, 0.0]
        coords[10, 6] = [80.0, 0.0]
        outliers = detect_speed_outliers(coords, valid, mad_k=6.0)
        self.assertTrue(outliers[10, 6])
        self.assertFalse(outliers[9, 6])
        self.assertFalse(outliers[11, 6])

    def test_one_euro_tracks_step_without_nan(self):
        values = np.zeros((30, 2))
        values[15:] = 10.0
        filtered = filter_sequence_one_euro(values, fps=30.0, min_cutoff=1.0, beta=0.007)
        self.assertFalse(np.any(np.isnan(filtered)))
        self.assertLess(abs(filtered[16, 0]), 10.0)
        self.assertGreater(filtered[-1, 0], 8.0)

    def test_refine_2d_end_to_end(self):
        keypoints = np.ones((16, 17, 3), dtype=np.float32)
        for t in range(16):
            keypoints[t, :, 0] = t
            keypoints[t, :, 1] = 0
        keypoints[5, 13, :2] = [90, 40]
        keypoints[6:9, 13, 2] = 0.05
        refined, stats = refine_2d_keypoints(keypoints, fps=30.0)
        self.assertEqual(refined.shape, keypoints.shape)
        self.assertGreater(stats["repaired_count"] + stats["speed_outlier_count"], 0)
        self.assertFalse(np.any(np.isnan(refined)))


class Test3DConstraints(unittest.TestCase):
    def test_bone_length_projection(self):
        pose = _standing_pose(100.0)
        targets = {bone_key(p, c): 50.0 for c, p in H36M_TOPOLOGY.items()}
        projected = project_bone_lengths(pose, targets, weight=1.0)
        np.testing.assert_allclose(projected[0], 0.0, atol=1e-8)
        for child, parent in H36M_TOPOLOGY.items():
            length = np.linalg.norm(projected[child] - projected[parent])
            self.assertAlmostEqual(length, 50.0, places=5)

    def test_length_symmetry(self):
        targets = {
            "4_5": 80.0,
            "1_2": 120.0,
        }
        out = apply_length_symmetry(targets, weight=1.0)
        self.assertAlmostEqual(out["4_5"], out["1_2"])
        self.assertAlmostEqual(out["4_5"], 100.0)

    def test_median_lengths_ignore_near_zero(self):
        poses = np.stack([_standing_pose(100.0) for _ in range(10)])
        poses[0, 3] = poses[0, 2]
        med = median_bone_lengths(poses, min_reliable_frames=3)
        self.assertGreater(med["2_3"], 1.0)

    def test_standard_ratio_fallback_for_collapsed_bone(self):
        targets = {"0_7": 43.3722, "2_3": 0.0}
        filled = fill_targets_with_standard_ratios(targets, min_length=1e-3)
        self.assertGreater(filled["2_3"], 1.0)
        self.assertAlmostEqual(filled["0_7"], 43.3722, places=3)

    def test_knee_overbend_is_opened(self):
        pose = _standing_pose(100.0)
        pose[3] = pose[1]  # ankle pulled to hip -> collapsed knee
        fixed = clamp_hinge_angle(pose, 1, 2, 3, min_deg=40.0, max_deg=178.0, weight=1.0)
        u = fixed[1] - fixed[2]
        w = fixed[3] - fixed[2]
        deg = np.degrees(np.arccos(np.clip(np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w)), -1, 1)))
        self.assertGreaterEqual(deg, 39.0)

    def test_refine_3d_reduces_bone_cv(self):
        poses = np.stack([_standing_pose(100.0) for _ in range(12)])
        poses[4, 3] = poses[4, 3] + np.array([0, 40, 0])
        poses[7, 6] = poses[7, 6] + np.array([0, -35, 0])
        refined, stats = refine_3d_poses(poses, fps=30.0)
        self.assertEqual(refined.shape, poses.shape)
        np.testing.assert_allclose(refined[:, 0], 0.0, atol=1e-5)
        self.assertLess(stats["bone_cv_after"], stats["bone_cv_before"])
        cfg = default_refinement_config()["3d"]["angle"]
        self.assertLessEqual(
            count_angle_violations(refined, cfg),
            count_angle_violations(poses, cfg),
        )

    def test_compare_report(self):
        poses = np.stack([_standing_pose(80.0) for _ in range(8)])
        noisy = poses.copy()
        noisy[3, 16] += 12
        report = compare_pose_sequences(noisy, poses, fps=30.0, mode="3d")
        self.assertIn("bone_cv_before", report)
        self.assertGreaterEqual(report["mean_correction"], 0.0)

    def test_compare_2d_counts_low_score(self):
        before = np.ones((6, 17, 3), dtype=np.float64)
        after = before.copy()
        before[2, 13, 2] = 0.05
        after[2, 13, :2] = [3.0, 4.0]
        after[2, 13, 2] = 0.4
        report = compare_pose_sequences(before, after, fps=30.0, mode="2d")
        self.assertEqual(report["low_score_before"], 1)
        self.assertEqual(report["low_score_after"], 0)
        self.assertGreaterEqual(report["changed_count"], 1)


if __name__ == "__main__":
    unittest.main()
