"""扩展 39 点协议与 RTMW3D 融合测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.extended_pose_fusion import (  # noqa: E402
    apply_similarity,
    fuse_extended_sequence,
    umeyama,
)
from utils.extended_pose_schema import (  # noqa: E402
    EXTENDED_JOINT_NAMES,
    EXTENDED_PARENTS,
    FACE_INDICES,
    FOOT_INDICES,
    HAND_INDICES,
    NUM_CORE_JOINTS,
    NUM_EXTENDED_JOINTS,
    NUM_WHOLEBODY_JOINTS,
    SHARED_ANCHORS,
    extract_extended_from_wholebody,
    is_core_pose,
    is_extended_pose,
    joint_index,
)


def _standing_h36m(scale: float = 100.0, facing: float = 40.0) -> np.ndarray:
    pose = np.zeros((17, 3), dtype=np.float64)
    pose[1] = [scale * 0.12, 0, 0]
    pose[2] = [scale * 0.12, scale * 0.45, 0]
    pose[3] = [scale * 0.12, scale * 0.90, 0]
    pose[4] = [-scale * 0.12, 0, 0]
    pose[5] = [-scale * 0.12, scale * 0.45, 0]
    pose[6] = [-scale * 0.12, scale * 0.90, 0]
    pose[7] = [0, -scale * 0.20, 0]
    pose[8] = [0, -scale * 0.40, 0]
    pose[9] = [0, -scale * 0.52, facing * 0.15]
    pose[10] = [0, -scale * 0.68, facing * 0.10]
    pose[11] = [-scale * 0.22, -scale * 0.40, 0]
    pose[12] = [-scale * 0.22, -scale * 0.18, 0]
    pose[13] = [-scale * 0.22, 0.02 * scale, 0]
    pose[14] = [scale * 0.22, -scale * 0.40, 0]
    pose[15] = [scale * 0.22, -scale * 0.18, 0]
    pose[16] = [scale * 0.22, 0.02 * scale, 0]
    return pose


def _wholebody_from_h36m(core: np.ndarray, scale: float = 1.4, translation=None) -> Tuple[np.ndarray, np.ndarray]:
    wb = np.zeros((NUM_WHOLEBODY_JOINTS, 3), dtype=np.float64)
    scores = np.ones((NUM_WHOLEBODY_JOINTS,), dtype=np.float64)
    for core_index, wb_index in SHARED_ANCHORS:
        wb[wb_index] = core[core_index] * scale
    hips = (core[1] + core[4]) * 0.5
    wb[11] = core[4] * scale
    wb[12] = core[1] * scale
    # feet ahead of ankles
    wb[17] = core[6] * scale + np.array([0.0, 8.0, 18.0])
    wb[18] = core[6] * scale + np.array([-6.0, 8.0, 14.0])
    wb[19] = core[6] * scale + np.array([0.0, 8.0, -8.0])
    wb[20] = core[3] * scale + np.array([0.0, 8.0, 18.0])
    wb[21] = core[3] * scale + np.array([6.0, 8.0, 14.0])
    wb[22] = core[3] * scale + np.array([0.0, 8.0, -8.0])
    for offset, src in enumerate([95, 99, 103, 107, 111]):
        wb[src] = core[13] * scale + np.array([-4.0 + offset, 6.0, 10.0 + offset])
    for offset, src in enumerate([116, 120, 124, 128, 132]):
        wb[src] = core[16] * scale + np.array([-4.0 + offset, 6.0, 10.0 + offset])
    # face 68-point clusters
    for idx in range(65, 71):
        wb[idx] = core[9] * scale + np.array([-6.0, -8.0, 4.0])
    for idx in range(59, 65):
        wb[idx] = core[9] * scale + np.array([6.0, -8.0, 4.0])
    wb[3] = core[9] * scale + np.array([-10.0, -4.0, -2.0])
    wb[4] = core[9] * scale + np.array([10.0, -4.0, -2.0])
    wb[53] = core[9] * scale + np.array([0.0, -2.0, 8.0])
    wb[74] = core[9] * scale + np.array([0.0, 4.0, 6.0])
    wb[80] = core[9] * scale + np.array([0.0, 8.0, 5.0])
    if translation is not None:
        wb = wb + np.asarray(translation, dtype=np.float64)
    return wb, scores


class TestSchema(unittest.TestCase):
    def test_counts_and_parents(self):
        self.assertEqual(len(EXTENDED_JOINT_NAMES), NUM_EXTENDED_JOINTS)
        self.assertEqual(len(EXTENDED_PARENTS), NUM_EXTENDED_JOINTS)
        self.assertEqual(EXTENDED_PARENTS[0], -1)
        self.assertEqual(EXTENDED_PARENTS[joint_index("left_big_toe")], joint_index("left_ankle"))
        self.assertEqual(EXTENDED_PARENTS[joint_index("right_thumb_tip")], joint_index("right_wrist"))
        self.assertEqual(len(FOOT_INDICES), 6)
        self.assertEqual(len(HAND_INDICES), 10)
        self.assertEqual(len(FACE_INDICES), 6)
        self.assertTrue(is_core_pose(np.zeros((4, 17, 3))))
        self.assertTrue(is_extended_pose(np.zeros((4, 39, 3))))
        self.assertFalse(is_extended_pose(np.zeros((4, 17, 3))))

    def test_wholebody_extraction_and_fallback(self):
        core = _standing_h36m()
        wb, scores = _wholebody_from_h36m(core)
        coords, out_scores, valid = extract_extended_from_wholebody(wb, scores, score_threshold=0.2)
        self.assertEqual(coords.shape, (39, 3))
        self.assertTrue(np.all(valid[17:]))
        self.assertFalse(np.any(valid[:17]))
        np.testing.assert_allclose(coords[17], wb[17], atol=1e-6)
        # fallback: wipe 68-point left eye, keep COCO left_eye
        wb_fb = wb.copy()
        scores_fb = scores.copy()
        scores_fb[65:71] = 0.0
        wb_fb[1] = np.array([1.0, 2.0, 3.0])
        coords_fb, _, valid_fb = extract_extended_from_wholebody(wb_fb, scores_fb, 0.2)
        self.assertTrue(valid_fb[joint_index("left_eye")])
        np.testing.assert_allclose(coords_fb[joint_index("left_eye")], [1.0, 2.0, 3.0])


class TestFusion(unittest.TestCase):
    def test_umeyama_recovers_similarity(self):
        rng = np.random.default_rng(0)
        src = rng.normal(size=(12, 3))
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        scale = 2.5
        translation = np.array([11.0, -4.0, 7.0])
        dst = apply_similarity(src, rotation, scale, translation)
        est_r, est_s, est_t = umeyama(src, dst)
        aligned = apply_similarity(src, est_r, est_s, est_t)
        np.testing.assert_allclose(aligned, dst, atol=1e-6)
        self.assertAlmostEqual(est_s, scale, places=6)

    def test_core_joints_unchanged_and_local_attach(self):
        core = np.stack([_standing_h36m() for _ in range(16)], axis=0)
        whole = []
        scores = []
        for t, pose in enumerate(core):
            wb, sc = _wholebody_from_h36m(pose, scale=1.7, translation=[30.0, -12.0, 8.0])
            whole.append(wb)
            scores.append(sc)
        fused = fuse_extended_sequence(core, np.stack(whole), np.stack(scores), fps=30.0)
        np.testing.assert_allclose(fused["core_pose_3d"], core, atol=1e-5)
        np.testing.assert_allclose(fused["extended_pose_3d"][:, :17], core, atol=1e-5)
        self.assertEqual(fused["wholebody_pose_3d_aligned"].shape, (16, 133, 3))
        self.assertEqual(fused["wholebody_valid"].shape, (16, 133))
        self.assertTrue(np.all(fused["wholebody_valid"]))
        self.assertGreater(fused["stats"]["accepted_frames"], 10)
        left_ankle = fused["extended_pose_3d"][:, 6]
        left_toe = fused["extended_pose_3d"][:, 17]
        offset = left_toe - left_ankle
        # 脚尖应在脚踝前方（+z）且贴在 MAG 脚踝上
        self.assertGreater(float(np.median(offset[:, 2])), 5.0)
        self.assertTrue(np.all(fused["extended_valid"][:, 17:23]))

    def test_low_confidence_is_hidden_not_hallucinated(self):
        core = np.stack([_standing_h36m() for _ in range(8)], axis=0)
        whole = []
        scores = []
        for pose in core:
            wb, sc = _wholebody_from_h36m(pose)
            sc[17:23] = 0.01
            whole.append(wb)
            scores.append(sc)
        fused = fuse_extended_sequence(core, np.stack(whole), np.stack(scores), fps=30.0)
        self.assertFalse(np.any(fused["extended_valid"][:, 17:23]))
        self.assertTrue(np.all(np.isnan(fused["extended_pose_3d"][:, 17:23])))

    def test_short_gap_interpolation(self):
        core = np.stack([_standing_h36m() for _ in range(12)], axis=0)
        whole = []
        scores = []
        for t, pose in enumerate(core):
            wb, sc = _wholebody_from_h36m(pose)
            if 4 <= t <= 6:
                sc[17] = 0.01
            whole.append(wb)
            scores.append(sc)
        fused = fuse_extended_sequence(core, np.stack(whole), np.stack(scores), fps=30.0)
        self.assertTrue(fused["extended_valid"][5, 17])
        self.assertFalse(np.isnan(fused["extended_pose_3d"][5, 17, 0]))

    def test_fuse_script_keeps_legacy_17_key(self):
        from utils.extended_pose_fusion import build_extended_npz_payload

        core = np.stack([_standing_h36m() for _ in range(6)], axis=0)
        whole = np.stack([_wholebody_from_h36m(pose)[0] for pose in core])
        scores = np.stack([_wholebody_from_h36m(pose)[1] for pose in core])
        fused = fuse_extended_sequence(core, whole, scores, fps=30.0)
        source = {
            "pred3d_root_relative_image_units": core.astype(np.float32),
            "image_width": np.asarray(1920, dtype=np.int32),
        }

        class _Wrapper(dict):
            @property
            def files(self):
                return list(self.keys())

        payload = build_extended_npz_payload(_Wrapper(source), fused)
        np.testing.assert_allclose(payload["pred3d_root_relative_image_units"], core, atol=1e-5)
        self.assertEqual(payload["extended_pose_3d"].shape, (6, 39, 3))
        self.assertEqual(payload["wholebody_pose_3d_aligned"].shape, (6, 133, 3))
        self.assertEqual(payload["wholebody_scores"].shape, (6, 133))
        self.assertEqual(payload["wholebody_valid"].shape, (6, 133))
        self.assertIn("extended_joint_names", payload)


if __name__ == "__main__":
    unittest.main()
