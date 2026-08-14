"""
2D / 3D 人体骨架精修与质量指标。

函数均为纯 NumPy 接口，不依赖视频读写或命令行。
关节顺序与视频流水线 H36M-17 一致：
root, rHip, rKnee, rAnkle, lHip, lKnee, lAnkle, spine, thorax, nose, head,
lShoulder, lElbow, lWrist, rShoulder, rElbow, rWrist
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


H36M_NAMES = (
    "root",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "spine",
    "thorax",
    "nose",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
)

# child -> parent
H36M_TOPOLOGY = {
    7: 0, 8: 7, 9: 8, 10: 9,
    4: 0, 5: 4, 6: 5,
    1: 0, 2: 1, 3: 2,
    11: 8, 12: 11, 13: 12,
    14: 8, 15: 14, 16: 15,
}

BONE_PROCESSING_ORDER = (
    1, 4, 7,
    2, 5, 8,
    3, 6, 9, 11, 14,
    10, 12, 15,
    13, 16,
)

H36M_PAIRS = (
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
)

# (left_parent, left_child, right_parent, right_child)
SYMMETRIC_BONES = (
    (0, 4, 0, 1),      # hip -> thigh root
    (4, 5, 1, 2),      # thigh
    (5, 6, 2, 3),      # shin
    (8, 11, 8, 14),    # shoulder
    (11, 12, 14, 15),  # upper arm
    (12, 13, 15, 16),  # forearm
)

# (parent, joint, child, min_deg_key, max_deg_key)
HINGE_JOINTS = (
    (1, 2, 3, "knee_min_deg", "knee_max_deg"),    # right knee
    (4, 5, 6, "knee_min_deg", "knee_max_deg"),    # left knee
    (14, 15, 16, "elbow_min_deg", "elbow_max_deg"),
    (11, 12, 13, "elbow_min_deg", "elbow_max_deg"),
)

_EPS = 1e-8


def load_refinement_config(path: Optional[str] = None) -> dict:
    """加载精修配置；path 为 None 时返回内置默认值。"""
    defaults = default_refinement_config()
    if not path:
        return defaults
    import json
    from pathlib import Path

    with Path(path).open("r", encoding="utf-8") as f:
        user_cfg = json.load(f)
    return _deep_merge(defaults, user_cfg)


def default_refinement_config() -> dict:
    return {
        "fps": 30.0,
        "2d": {
            "score_threshold": 0.25,
            "speed_mad_k": 6.0,
            "max_gap": 8,
            "repaired_score": 0.4,
            "one_euro": {
                "enabled": True,
                "min_cutoff": 1.0,
                "beta": 0.007,
                "dcutoff": 1.0,
            },
        },
        "3d": {
            "smooth": {
                "enabled": True,
                "min_cutoff": 1.2,
                "beta": 0.01,
                "dcutoff": 1.0,
            },
            "bone": {
                "iterations": 3,
                "length_weight": 0.85,
                "symmetry_weight": 0.35,
                "min_reliable_frames": 30,
                "min_length": 1e-3,
            },
            "angle": {
                "enabled": True,
                "weight": 0.6,
                "knee_min_deg": 25.0,
                "knee_max_deg": 178.0,
                "elbow_min_deg": 20.0,
                "elbow_max_deg": 178.0,
            },
        },
    }


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def bone_key(parent: int, child: int) -> str:
    return f"{parent}_{child}"


def bone_lengths(pose: np.ndarray) -> Dict[str, float]:
    """单帧各骨段长度。pose: (17, D)。"""
    lengths = {}
    for child, parent in H36M_TOPOLOGY.items():
        lengths[bone_key(parent, child)] = float(np.linalg.norm(pose[child] - pose[parent]))
    return lengths


def sequence_bone_lengths(poses: np.ndarray) -> np.ndarray:
    """poses: (T, 17, D) -> (T, n_bones) 按 BONE_PROCESSING_ORDER。"""
    lengths = np.zeros((poses.shape[0], len(BONE_PROCESSING_ORDER)), dtype=np.float64)
    for i, child in enumerate(BONE_PROCESSING_ORDER):
        parent = H36M_TOPOLOGY[child]
        lengths[:, i] = np.linalg.norm(poses[:, child] - poses[:, parent], axis=-1)
    return lengths


def standard_bone_ratios() -> Dict[str, float]:
    """读取 config.STANDARD_BONE_RATIOS；失败时返回空字典。"""
    try:
        from config import STANDARD_BONE_RATIOS

        return dict(STANDARD_BONE_RATIOS)
    except Exception:
        return {}


def fill_targets_with_standard_ratios(
    targets: Dict[str, float],
    min_length: float = 1e-3,
) -> Dict[str, float]:
    """
    仅当某骨段中位长度不可靠（接近 0）时，用标准比例按当前视频尺度补全。
    默认仍以视频内中位数为主，避免把全局人体比例强加给单个运动员。
    """
    ratios = standard_bone_ratios()
    if not ratios:
        return dict(targets)
    result = dict(targets)
    ref_key = "0_7"
    ref_ratio = float(ratios.get(ref_key, 0.0))
    ref_len = float(result.get(ref_key, 0.0))
    if ref_len >= min_length and ref_ratio > 0:
        scale = ref_len / ref_ratio
    else:
        scales = []
        for key, length in result.items():
            ratio = float(ratios.get(key, 0.0))
            if length >= min_length and ratio > 0:
                scales.append(length / ratio)
        scale = float(np.median(scales)) if scales else 0.0
    if scale <= 0:
        return result
    for key, ratio in ratios.items():
        if float(result.get(key, 0.0)) < min_length:
            result[key] = float(ratio) * scale
    return result


def median_bone_lengths(
    poses: np.ndarray,
    valid: Optional[np.ndarray] = None,
    min_length: float = 1e-3,
    min_reliable_frames: int = 30,
) -> Dict[str, float]:
    """
    从可靠帧估计每条骨骼的中位长度。
    valid: 可选 (T, 17) bool，True 表示该关节可用。
    个别骨段若全程塌缩，再用 STANDARD_BONE_RATIOS 按当前视频尺度补全。
    """
    t_count = poses.shape[0]
    targets: Dict[str, float] = {}
    for child, parent in H36M_TOPOLOGY.items():
        vec = poses[:, child] - poses[:, parent]
        length = np.linalg.norm(vec, axis=-1)
        mask = length >= min_length
        if valid is not None:
            mask &= valid[:, child] & valid[:, parent]
        if int(np.sum(mask)) < max(3, min(min_reliable_frames, t_count)):
            mask = length >= min_length
        if not np.any(mask):
            targets[bone_key(parent, child)] = float(np.median(length)) if t_count else 0.0
        else:
            targets[bone_key(parent, child)] = float(np.median(length[mask]))
    return fill_targets_with_standard_ratios(targets, min_length=min_length)


def apply_length_symmetry(
    targets: Dict[str, float],
    weight: float,
) -> Dict[str, float]:
    """左右对应骨段长度向均值靠拢，不改变左右姿态。"""
    result = dict(targets)
    weight = float(np.clip(weight, 0.0, 1.0))
    if weight <= 0:
        return result
    for lp, lc, rp, rc in SYMMETRIC_BONES:
        lk = bone_key(lp, lc)
        rk = bone_key(rp, rc)
        left = result.get(lk, 0.0)
        right = result.get(rk, 0.0)
        mean = 0.5 * (left + right)
        result[lk] = (1.0 - weight) * left + weight * mean
        result[rk] = (1.0 - weight) * right + weight * mean
    return result


def _smoothing_factor(dt: float, cutoff: float) -> float:
    tau = 1.0 / (2.0 * np.pi * max(cutoff, _EPS))
    return 1.0 / (1.0 + tau / max(dt, _EPS))


class OneEuroFilter:
    """Casiez et al. One Euro Filter，用于 2D/3D 坐标去抖。"""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, dcutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self._hat_x: Optional[np.ndarray] = None
        self._hat_dx: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._hat_x = None
        self._hat_dx = None

    def apply(self, x: np.ndarray, dt: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._hat_x is None:
            self._hat_x = x.copy()
            self._hat_dx = np.zeros_like(x)
            return x.copy()
        dx = (x - self._hat_x) / max(dt, _EPS)
        a_d = _smoothing_factor(dt, self.dcutoff)
        self._hat_dx = a_d * dx + (1.0 - a_d) * self._hat_dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._hat_dx))
        a = _smoothing_factor(dt, cutoff)
        self._hat_x = a * x + (1.0 - a) * self._hat_x
        return self._hat_x.copy()


def filter_sequence_one_euro(
    values: np.ndarray,
    valid: Optional[np.ndarray] = None,
    fps: float = 30.0,
    min_cutoff: float = 1.0,
    beta: float = 0.007,
    dcutoff: float = 1.0,
) -> np.ndarray:
    """
    对 (T, ...) 序列按时间做 One Euro。
    valid: 可选 (T,) 或与 values 广播兼容的 mask；无效帧保持原值且不更新滤波器状态。
    """
    values = np.asarray(values, dtype=np.float64)
    out = values.copy()
    dt = 1.0 / max(float(fps), _EPS)
    if valid is None:
        valid = np.ones(values.shape[0], dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)
        if valid.ndim > 1:
            valid = np.all(valid.reshape(valid.shape[0], -1), axis=1)

    filt = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, dcutoff=dcutoff)
    for t in range(values.shape[0]):
        if not valid[t]:
            filt.reset()
            continue
        out[t] = filt.apply(values[t], dt)
    return out.astype(values.dtype, copy=False)


def detect_low_score(scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(scores, dtype=np.float64) < float(threshold)


def detect_speed_outliers(
    coords: np.ndarray,
    valid: np.ndarray,
    mad_k: float = 6.0,
) -> np.ndarray:
    """
    用速度的 MAD 检测跳点。coords: (T, 17, D)，valid: (T, 17)。
    仅当某一帧的前后两段位移都异常时才标记该帧，避免把尖峰后的正常帧一起删掉。
    """
    t_count, n_joints = coords.shape[:2]
    outlier = np.zeros((t_count, n_joints), dtype=bool)
    if t_count < 3:
        return outlier
    delta = coords[1:] - coords[:-1]
    speed = np.linalg.norm(delta, axis=-1)
    pair_valid = valid[1:] & valid[:-1]
    for j in range(n_joints):
        samples = speed[:, j][pair_valid[:, j]]
        if samples.size < 8:
            continue
        median = float(np.median(samples))
        mad = float(np.median(np.abs(samples - median)))
        if mad > _EPS:
            scale = 1.4826 * mad
        else:
            # 匀速/静止序列的 MAD 为 0，不能回退到含离群点的标准差。
            scale = max(abs(median) * 0.25, 1.0)
        thresh = median + float(mad_k) * scale
        jump = pair_valid[:, j] & (speed[:, j] > thresh)
        outlier[1:-1, j] |= jump[:-1] & jump[1:]
        if jump[0] and t_count >= 3 and pair_valid[1, j]:
            d0 = float(np.linalg.norm(coords[0, j] - coords[2, j]))
            d1 = float(np.linalg.norm(coords[1, j] - coords[2, j]))
            if d0 > d1:
                outlier[0, j] = True
        if jump[-1] and t_count >= 3 and pair_valid[-2, j]:
            d_last = float(np.linalg.norm(coords[-1, j] - coords[-3, j]))
            d_prev = float(np.linalg.norm(coords[-2, j] - coords[-3, j]))
            if d_last > d_prev:
                outlier[-1, j] = True
    return outlier


def interpolate_gaps(
    coords: np.ndarray,
    scores: np.ndarray,
    valid: np.ndarray,
    max_gap: int,
    repaired_score: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    对每个关节的短缺口做线性插值。
    返回 coords, scores, valid, repaired_mask。
    """
    coords = np.asarray(coords, dtype=np.float64).copy()
    scores = np.asarray(scores, dtype=np.float64).copy()
    valid = np.asarray(valid, dtype=bool).copy()
    repaired = np.zeros_like(valid)
    t_count, n_joints = valid.shape
    max_gap = int(max_gap)
    for j in range(n_joints):
        t = 0
        while t < t_count:
            if valid[t, j]:
                t += 1
                continue
            start = t
            while t < t_count and not valid[t, j]:
                t += 1
            end = t
            gap = end - start
            left = start - 1
            right = end
            if gap > max_gap or left < 0 or right >= t_count:
                continue
            if not valid[left, j] or not valid[right, j]:
                continue
            span = right - left
            for k, idx in enumerate(range(start, end)):
                alpha = (k + 1) / float(span)
                coords[idx, j] = (1.0 - alpha) * coords[left, j] + alpha * coords[right, j]
                scores[idx, j] = float(repaired_score)
                valid[idx, j] = True
                repaired[idx, j] = True
    return coords, scores, valid, repaired


def refine_2d_keypoints(
    keypoints: np.ndarray,
    cfg: Optional[dict] = None,
    fps: Optional[float] = None,
) -> Tuple[np.ndarray, dict]:
    """
    精修 H36M 2D 关键点。
    keypoints: (T, 17, 3) 为 x, y, score。
    返回 refined (T, 17, 3) 与统计信息。
    """
    cfg = cfg or default_refinement_config()
    two_d = cfg.get("2d", {})
    fps = float(fps if fps is not None else cfg.get("fps", 30.0))
    kps = np.asarray(keypoints, dtype=np.float64)
    if kps.ndim != 3 or kps.shape[1] != 17 or kps.shape[2] < 3:
        raise ValueError(f"期望 (T, 17, 3) H36M 关键点，实际 {kps.shape}")

    coords = kps[:, :, :2].copy()
    scores = kps[:, :, 2].copy()
    score_thr = float(two_d.get("score_threshold", 0.25))
    valid = ~detect_low_score(scores, score_thr)
    low_score_count = int(np.sum(~valid))

    outliers = detect_speed_outliers(
        coords, valid, mad_k=float(two_d.get("speed_mad_k", 6.0))
    )
    valid &= ~outliers
    outlier_count = int(np.sum(outliers))

    coords, scores, valid, repaired = interpolate_gaps(
        coords,
        scores,
        valid,
        max_gap=int(two_d.get("max_gap", 8)),
        repaired_score=float(two_d.get("repaired_score", 0.4)),
    )
    repaired_count = int(np.sum(repaired))

    euro = two_d.get("one_euro", {})
    if euro.get("enabled", True):
        filtered = coords.copy()
        for j in range(coords.shape[1]):
            filtered[:, j] = filter_sequence_one_euro(
                coords[:, j],
                valid=valid[:, j],
                fps=fps,
                min_cutoff=float(euro.get("min_cutoff", 1.0)),
                beta=float(euro.get("beta", 0.007)),
                dcutoff=float(euro.get("dcutoff", 1.0)),
            )
        coords = filtered

    refined = np.concatenate([coords, scores[:, :, None]], axis=2)
    stats = {
        "frames": int(kps.shape[0]),
        "low_score_count": low_score_count,
        "speed_outlier_count": outlier_count,
        "repaired_count": repaired_count,
        "still_invalid_count": int(np.sum(~valid)),
        "score_threshold": score_thr,
        "fps": fps,
    }
    return refined.astype(np.float32), stats


def project_bone_lengths(
    pose: np.ndarray,
    targets: Dict[str, float],
    weight: float = 1.0,
) -> np.ndarray:
    """保持方向、将骨长向目标长度投影。pose: (17, 3)。root 保持不变。"""
    out = np.asarray(pose, dtype=np.float64).copy()
    weight = float(np.clip(weight, 0.0, 1.0))
    out[0] = pose[0]
    for child in BONE_PROCESSING_ORDER:
        parent = H36M_TOPOLOGY[child]
        vec = out[child] - out[parent]
        current = float(np.linalg.norm(vec))
        target = float(targets.get(bone_key(parent, child), current))
        blended = (1.0 - weight) * current + weight * target
        if current < _EPS:
            out[child] = out[parent]
            continue
        out[child] = out[parent] + vec / current * blended
    return out


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < _EPS or abs(angle) < _EPS:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def _body_forward(pose: np.ndarray) -> np.ndarray:
    right = pose[1] - pose[4]
    up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    forward = np.cross(right, up)
    norm = np.linalg.norm(forward)
    if norm < _EPS:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return forward / norm


def clamp_hinge_angle(
    pose: np.ndarray,
    parent: int,
    joint: int,
    child: int,
    min_deg: float,
    max_deg: float,
    weight: float,
) -> np.ndarray:
    """软约束铰链角到 [min_deg, max_deg]，并轻微纠正过伸。"""
    out = np.asarray(pose, dtype=np.float64).copy()
    u = out[parent] - out[joint]
    w = out[child] - out[joint]
    n_u = float(np.linalg.norm(u))
    n_w = float(np.linalg.norm(w))
    if n_u < _EPS or n_w < _EPS:
        return out
    cosang = float(np.clip(np.dot(u, w) / (n_u * n_w), -1.0, 1.0))
    angle = float(np.arccos(cosang))
    min_rad = np.deg2rad(min_deg)
    max_rad = np.deg2rad(max_deg)
    target = float(np.clip(angle, min_rad, max_rad))

    axis = np.cross(u, w)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < _EPS:
        forward = _body_forward(out)
        axis = np.cross(u, forward)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < _EPS:
            return out
        if np.dot(w, forward) < 0 and angle > np.deg2rad(170.0):
            target = min(target, np.deg2rad(max_deg - 2.0))

    if abs(target - angle) < 1e-6:
        return out
    delta = (target - angle) * float(np.clip(weight, 0.0, 1.0))
    rot = _rotation_matrix(axis, delta)
    out[child] = out[joint] + rot @ w
    return out


def apply_angle_constraints(pose: np.ndarray, angle_cfg: dict) -> np.ndarray:
    out = np.asarray(pose, dtype=np.float64).copy()
    if not angle_cfg.get("enabled", True):
        return out
    weight = float(angle_cfg.get("weight", 0.6))
    for parent, joint, child, min_key, max_key in HINGE_JOINTS:
        out = clamp_hinge_angle(
            out,
            parent,
            joint,
            child,
            min_deg=float(angle_cfg.get(min_key, 20.0)),
            max_deg=float(angle_cfg.get(max_key, 178.0)),
            weight=weight,
        )
    return out


def refine_3d_poses(
    poses: np.ndarray,
    cfg: Optional[dict] = None,
    fps: Optional[float] = None,
    valid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, dict]:
    """
    精修 root-relative 3D 姿态。poses: (T, 17, 3)。
    顺序：时序平滑 → 骨长/对称/角度迭代投影。
    """
    cfg = cfg or default_refinement_config()
    three_d = cfg.get("3d", {})
    fps = float(fps if fps is not None else cfg.get("fps", 30.0))
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1] != 17 or poses.shape[2] != 3:
        raise ValueError(f"期望 (T, 17, 3) 3D 姿态，实际 {poses.shape}")

    before_lengths = sequence_bone_lengths(poses)
    refined = poses.copy()
    refined[:, 0, :] = 0.0

    smooth_cfg = three_d.get("smooth", {})
    if smooth_cfg.get("enabled", True):
        for j in range(1, 17):
            refined[:, j] = filter_sequence_one_euro(
                refined[:, j],
                fps=fps,
                min_cutoff=float(smooth_cfg.get("min_cutoff", 1.2)),
                beta=float(smooth_cfg.get("beta", 0.01)),
                dcutoff=float(smooth_cfg.get("dcutoff", 1.0)),
            )
        refined[:, 0, :] = 0.0

    bone_cfg = three_d.get("bone", {})
    targets = median_bone_lengths(
        refined,
        valid=valid,
        min_length=float(bone_cfg.get("min_length", 1e-3)),
        min_reliable_frames=int(bone_cfg.get("min_reliable_frames", 30)),
    )
    targets = apply_length_symmetry(targets, float(bone_cfg.get("symmetry_weight", 0.35)))
    length_weight = float(bone_cfg.get("length_weight", 0.85))
    iterations = int(bone_cfg.get("iterations", 3))
    angle_cfg = three_d.get("angle", {})

    for t in range(refined.shape[0]):
        pose = refined[t]
        for _ in range(max(1, iterations)):
            pose = project_bone_lengths(pose, targets, weight=length_weight)
            pose = apply_angle_constraints(pose, angle_cfg)
            pose[0] = 0.0
        refined[t] = pose

    after_lengths = sequence_bone_lengths(refined)
    correction = np.linalg.norm(refined - poses, axis=-1)
    stats = {
        "frames": int(poses.shape[0]),
        "fps": fps,
        "mean_joint_correction": float(np.mean(correction)),
        "max_joint_correction": float(np.max(correction)),
        "bone_cv_before": _bone_cv(before_lengths),
        "bone_cv_after": _bone_cv(after_lengths),
        "jerk_before": mean_jerk(poses, fps),
        "jerk_after": mean_jerk(refined, fps),
        "target_bone_lengths": targets,
    }
    return refined.astype(np.float32), stats


def _bone_cv(lengths: np.ndarray) -> float:
    means = np.mean(lengths, axis=0)
    stds = np.std(lengths, axis=0)
    ratios = []
    for mean, std in zip(means, stds):
        if mean > _EPS:
            ratios.append(std / mean)
    return float(np.mean(ratios)) if ratios else 0.0


def mean_jerk(poses: np.ndarray, fps: float) -> float:
    """平均关节 jerk（三阶差分幅度）。"""
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape[0] < 4:
        return 0.0
    dt = 1.0 / max(float(fps), _EPS)
    acc = np.diff(poses, n=3, axis=0) / (dt ** 3)
    return float(np.mean(np.linalg.norm(acc, axis=-1)))


def hinge_angles_deg(pose: np.ndarray) -> Dict[str, float]:
    result = {}
    names = {
        2: "right_knee",
        5: "left_knee",
        15: "right_elbow",
        12: "left_elbow",
    }
    for parent, joint, child, _, _ in HINGE_JOINTS:
        u = pose[parent] - pose[joint]
        w = pose[child] - pose[joint]
        n_u = np.linalg.norm(u)
        n_w = np.linalg.norm(w)
        if n_u < _EPS or n_w < _EPS:
            result[names[joint]] = float("nan")
            continue
        cosang = float(np.clip(np.dot(u, w) / (n_u * n_w), -1.0, 1.0))
        result[names[joint]] = float(np.degrees(np.arccos(cosang)))
    return result


def count_angle_violations(poses: np.ndarray, angle_cfg: Optional[dict] = None) -> int:
    angle_cfg = angle_cfg or default_refinement_config()["3d"]["angle"]
    count = 0
    for pose in poses:
        for parent, joint, child, min_key, max_key in HINGE_JOINTS:
            u = pose[parent] - pose[joint]
            w = pose[child] - pose[joint]
            n_u = np.linalg.norm(u)
            n_w = np.linalg.norm(w)
            if n_u < _EPS or n_w < _EPS:
                continue
            cosang = float(np.clip(np.dot(u, w) / (n_u * n_w), -1.0, 1.0))
            deg = float(np.degrees(np.arccos(cosang)))
            lo = float(angle_cfg.get(min_key, 20.0))
            hi = float(angle_cfg.get(max_key, 178.0))
            if deg < lo or deg > hi:
                count += 1
    return count


def compare_pose_sequences(
    before: np.ndarray,
    after: np.ndarray,
    fps: float = 30.0,
    mode: str = "3d",
) -> dict:
    """比较精修前后序列，用于质量报告。"""
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    if before.shape != after.shape:
        raise ValueError(f"形状不一致: {before.shape} vs {after.shape}")
    coords_before = before[..., :3] if mode == "3d" else before[..., :2]
    coords_after = after[..., :3] if mode == "3d" else after[..., :2]
    delta = np.linalg.norm(coords_after - coords_before, axis=-1)
    report = {
        "mode": mode,
        "frames": int(before.shape[0]),
        "mean_correction": float(np.mean(delta)),
        "max_correction": float(np.max(delta)),
        "jerk_before": mean_jerk(coords_before, fps),
        "jerk_after": mean_jerk(coords_after, fps),
        "bone_cv_before": _bone_cv(sequence_bone_lengths(coords_before)),
        "bone_cv_after": _bone_cv(sequence_bone_lengths(coords_after)),
    }
    if mode == "3d":
        cfg = default_refinement_config()["3d"]["angle"]
        report["angle_violations_before"] = count_angle_violations(coords_before, cfg)
        report["angle_violations_after"] = count_angle_violations(coords_after, cfg)
    if mode == "2d" and before.shape[-1] >= 3:
        score_thr = float(default_refinement_config()["2d"]["score_threshold"])
        report["mean_score_before"] = float(np.mean(before[..., 2]))
        report["mean_score_after"] = float(np.mean(after[..., 2]))
        report["low_score_before"] = int(np.sum(before[..., 2] < score_thr))
        report["low_score_after"] = int(np.sum(after[..., 2] < score_thr))
        report["changed_count"] = int(np.sum(delta > 1e-3))
    return report
