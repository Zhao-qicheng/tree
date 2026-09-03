"""把 RTMW3D 133 点对齐并附着到 MotionAGFormer H36M-17 核心骨架。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from utils.extended_pose_schema import (
    ATTACHMENT_GROUPS,
    EXTENDED_JOINT_NAMES,
    NUM_CORE_JOINTS,
    NUM_EXTENDED_JOINTS,
    NUM_WHOLEBODY_JOINTS,
    SCHEMA_VERSION,
    SOURCE_STATUS_CORE,
    SOURCE_STATUS_EXTENDED,
    SOURCE_STATUS_HIDDEN,
    SOURCE_STATUS_INTERPOLATED,
    SOURCE_STATUS_REJECTED,
    default_extended_config,
    extract_extended_from_wholebody,
    pelvis_point,
    schema_metadata,
    shared_anchor_points,
)
from utils.pose_constraints import filter_sequence_one_euro, interpolate_gaps

_EPS = 1e-8


def load_extended_config(path: Optional[str] = None) -> dict:
    cfg = default_extended_config()
    if not path:
        return cfg
    with Path(path).open("r", encoding="utf-8") as handle:
        user_cfg = json.load(handle)
    merged = dict(cfg)
    for key, value in user_cfg.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def umeyama(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """估计 source -> target 的相似变换 (R, scale, t)。"""
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"umeyama 期望 (N, 3) 对应点，实际 {src.shape} vs {dst.shape}")
    if src.shape[0] < 3:
        raise ValueError("umeyama 至少需要 3 个对应点")

    n_points = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    covariance = (src_c.T @ dst_c) / float(n_points)
    u_mat, singular, vt_mat = np.linalg.svd(covariance)
    rotation = vt_mat.T @ u_mat.T
    if np.linalg.det(rotation) < 0:
        vt_mat = vt_mat.copy()
        vt_mat[-1] *= -1.0
        rotation = vt_mat.T @ u_mat.T
        singular = singular.copy()
        singular[-1] *= -1.0
    src_var = float(np.sum(src_c ** 2) / float(n_points))
    scale = float(np.sum(singular) / max(src_var, _EPS))
    translation = mu_dst - scale * (rotation @ mu_src)
    return rotation, scale, translation


def apply_similarity(
    points: np.ndarray,
    rotation: np.ndarray,
    scale: float,
    translation: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    original_shape = pts.shape
    flat = pts.reshape(-1, 3)
    transformed = (scale * (flat @ rotation.T)) + translation
    return transformed.reshape(original_shape)


def alignment_residual(
    source: np.ndarray,
    target: np.ndarray,
    rotation: np.ndarray,
    scale: float,
    translation: np.ndarray,
) -> float:
    aligned = apply_similarity(source, rotation, scale, translation)
    delta = np.linalg.norm(aligned - target, axis=-1)
    body_scale = float(np.median(np.linalg.norm(target - target.mean(axis=0), axis=-1)))
    if body_scale <= _EPS:
        body_scale = 1.0
    return float(np.mean(delta) / body_scale)


def _valid_mask(points: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(points), axis=-1)


def estimate_global_similarity(
    core_poses: np.ndarray,
    wholebody_poses: np.ndarray,
    wholebody_scores: np.ndarray,
    cfg: Optional[dict] = None,
    sample_limit: int = 240,
) -> Tuple[np.ndarray, float, dict]:
    """用多帧共用锚点估计全局旋转和基准尺度。"""
    cfg = cfg or default_extended_config()
    score_thr = float(cfg.get("score_threshold", 0.25))
    min_anchors = int(cfg.get("shared_anchor_min", 6))
    core = np.asarray(core_poses, dtype=np.float64)
    whole = np.asarray(wholebody_poses, dtype=np.float64)
    scores = np.asarray(wholebody_scores, dtype=np.float64)

    t_count = core.shape[0]
    if t_count <= sample_limit:
        frame_indices = np.arange(t_count)
    else:
        frame_indices = np.unique(np.round(np.linspace(0, t_count - 1, sample_limit)).astype(int))

    src_all = []
    dst_all = []
    used_frames = 0
    for t in frame_indices:
        src, dst, _ = shared_anchor_points(core[t], whole[t], scores[t], score_thr)
        if src.shape[0] < min_anchors:
            continue
        src_all.append(src)
        dst_all.append(dst)
        used_frames += 1
    if not src_all:
        raise ValueError("没有足够的共用锚点来估计 RTMW3D -> H36M 相似变换")

    src_stack = np.concatenate(src_all, axis=0)
    dst_stack = np.concatenate(dst_all, axis=0)
    rotation, scale, _translation = umeyama(src_stack, dst_stack)
    residual = alignment_residual(src_stack, dst_stack, rotation, scale, dst_stack.mean(axis=0) - scale * (rotation @ src_stack.mean(axis=0)))
    info = {
        "used_frames": int(used_frames),
        "used_points": int(src_stack.shape[0]),
        "global_scale": float(scale),
        "global_residual": float(residual),
    }
    return rotation, float(scale), info


def _window_median(values: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    t_count = values.shape[0]
    half = max(1, int(window) // 2)
    out = np.full(t_count, np.nan, dtype=np.float64)
    for t in range(t_count):
        start = max(0, t - half)
        end = min(t_count, t + half + 1)
        samples = values[start:end][valid[start:end]]
        if samples.size:
            out[t] = float(np.median(samples))
    # 填补窗口空洞
    finite = np.isfinite(out)
    if finite.any():
        first = int(np.argmax(finite))
        out[:first] = out[first]
        last_value = out[first]
        for t in range(first, t_count):
            if np.isfinite(out[t]):
                last_value = out[t]
            else:
                out[t] = last_value
    else:
        out[:] = 1.0
    return out


def _window_rotations(rotations: np.ndarray, valid: np.ndarray, window: int, fallback: np.ndarray) -> np.ndarray:
    """Average nearby rotations and project each result back onto SO(3)."""
    t_count = rotations.shape[0]
    half = max(1, int(window) // 2)
    out = np.empty((t_count, 3, 3), dtype=np.float64)
    for t in range(t_count):
        start = max(0, t - half)
        end = min(t_count, t + half + 1)
        samples = rotations[start:end][valid[start:end]]
        mean_rotation = samples.mean(axis=0) if samples.size else fallback
        u_mat, _singular, vt_mat = np.linalg.svd(mean_rotation)
        projected = u_mat @ vt_mat
        if np.linalg.det(projected) < 0:
            u_mat[:, -1] *= -1.0
            projected = u_mat @ vt_mat
        out[t] = projected
    return out


def _bone_length_scale(
    core_pose: np.ndarray,
    whole_pose: np.ndarray,
    rotation: np.ndarray,
    scores: np.ndarray,
    score_threshold: float,
) -> Optional[float]:
    src, dst, _ = shared_anchor_points(core_pose, whole_pose, scores, score_threshold)
    if src.shape[0] < 4:
        return None
    src_r = src @ rotation.T
    src_len = []
    dst_len = []
    for i in range(src_r.shape[0]):
        for j in range(i + 1, src_r.shape[0]):
            a = float(np.linalg.norm(src_r[i] - src_r[j]))
            b = float(np.linalg.norm(dst[i] - dst[j]))
            if a > _EPS and b > _EPS:
                src_len.append(a)
                dst_len.append(b)
    if len(src_len) < 3:
        return None
    ratios = np.asarray(dst_len, dtype=np.float64) / np.asarray(src_len, dtype=np.float64)
    return float(np.median(ratios))


def local_attach_frame(
    core_pose: np.ndarray,
    extras_aligned: np.ndarray,
    extras_valid: np.ndarray,
    parents_aligned: np.ndarray,
    length_medians: Dict[int, float],
    length_clip: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """把扩展点相对父节点的偏移接到 MotionAGFormer 父关节上。"""
    fused = extras_aligned.copy()
    valid = extras_valid.copy()
    low, high = float(length_clip[0]), float(length_clip[1])
    for group in ATTACHMENT_GROUPS:
        parent = int(group["parent"])
        mag_parent = np.asarray(core_pose[parent], dtype=np.float64)
        aligned_parent = np.asarray(parents_aligned[parent], dtype=np.float64)
        if not np.all(np.isfinite(mag_parent)) or not np.all(np.isfinite(aligned_parent)):
            for index in group["indices"]:
                valid[index] = False
            continue
        for index in group["indices"]:
            idx = int(index)
            if not valid[idx] or not np.all(np.isfinite(extras_aligned[idx])):
                valid[idx] = False
                continue
            offset = extras_aligned[idx] - aligned_parent
            length = float(np.linalg.norm(offset))
            median_len = float(length_medians.get(idx, length))
            if median_len > _EPS and (length < low * median_len or length > high * median_len):
                valid[idx] = False
                continue
            fused[idx] = mag_parent + offset
    return fused, valid


def fuse_extended_sequence(
    core_poses: np.ndarray,
    wholebody_poses: np.ndarray,
    wholebody_scores: np.ndarray,
    cfg: Optional[dict] = None,
    fps: Optional[float] = None,
) -> dict:
    """融合整段序列。核心 17 点原样保留，扩展点局部附着。"""
    cfg = cfg or default_extended_config()
    fps = float(fps if fps is not None else cfg.get("fps", 60.0))
    score_thr = float(cfg.get("score_threshold", 0.25))
    min_anchors = int(cfg.get("shared_anchor_min", 6))
    residual_thr = float(cfg.get("residual_threshold", 0.22))
    window = int(cfg.get("scale_window", 31))
    max_gap = int(cfg.get("max_gap", 8))
    repaired_score = float(cfg.get("repaired_score", 0.35))
    length_clip = cfg.get("length_clip", [0.25, 2.8])
    euro = cfg.get("one_euro", {})

    core = np.asarray(core_poses, dtype=np.float64)
    whole = np.asarray(wholebody_poses, dtype=np.float64)
    scores_wb = np.asarray(wholebody_scores, dtype=np.float64)
    if core.ndim != 3 or core.shape[1] != NUM_CORE_JOINTS:
        raise ValueError(f"期望核心姿态 (T, 17, 3)，实际 {core.shape}")
    if whole.ndim != 3 or whole.shape[1] != NUM_WHOLEBODY_JOINTS:
        raise ValueError(f"期望 WholeBody 姿态 (T, 133, 3)，实际 {whole.shape}")
    if core.shape[0] != whole.shape[0]:
        raise ValueError(f"帧数不一致: core={core.shape[0]} wholebody={whole.shape[0]}")

    t_count = core.shape[0]
    rotation, global_scale, global_info = estimate_global_similarity(core, whole, scores_wb, cfg)

    per_frame_scale = np.full(t_count, np.nan, dtype=np.float64)
    per_frame_residual = np.full(t_count, np.nan, dtype=np.float64)
    accepted = np.zeros(t_count, dtype=bool)
    translations = np.zeros((t_count, 3), dtype=np.float64)
    rotations = np.repeat(rotation[None, :, :], t_count, axis=0)
    aligned_core_from_wb = np.full((t_count, NUM_CORE_JOINTS, 3), np.nan, dtype=np.float64)

    for t in range(t_count):
        src, dst, _ = shared_anchor_points(core[t], whole[t], scores_wb[t], score_thr)
        if src.shape[0] >= min_anchors:
            rotation_t, scale_t, translation_t = umeyama(src, dst)
            rotations[t] = rotation_t
            translations[t] = translation_t
        else:
            scale_t = _bone_length_scale(core[t], whole[t], rotation, scores_wb[t], score_thr)
            if scale_t is None:
                scale_t = global_scale
        per_frame_scale[t] = scale_t
        pelvis_src = pelvis_point(whole[t], scores_wb[t])
        pelvis_dst = np.asarray(core[t, 0], dtype=np.float64)
        if src.shape[0] < min_anchors:
            if pelvis_src is None or not np.all(np.isfinite(pelvis_dst)):
                if src.shape[0]:
                    translations[t] = dst.mean(axis=0) - scale_t * (rotations[t] @ src.mean(axis=0))
                else:
                    translations[t] = 0.0
            else:
                translations[t] = pelvis_dst - scale_t * (rotations[t] @ pelvis_src)
        if src.shape[0] >= min_anchors:
            residual = alignment_residual(src, dst, rotations[t], scale_t, translations[t])
            per_frame_residual[t] = residual
            accepted[t] = residual <= residual_thr

    scale_valid = np.isfinite(per_frame_scale) & accepted
    if not np.any(scale_valid):
        scale_valid = np.isfinite(per_frame_scale)
    smooth_scale = _window_median(per_frame_scale, scale_valid, window)
    rotation_valid = np.all(np.isfinite(rotations), axis=(1, 2)) & accepted
    if not np.any(rotation_valid):
        rotation_valid = np.all(np.isfinite(rotations), axis=(1, 2))
    smooth_rotation = _window_rotations(rotations, rotation_valid, window, rotation)

    whole_aligned = np.full_like(whole[:, :, :3], np.nan)
    whole_valid = np.zeros((t_count, NUM_WHOLEBODY_JOINTS), dtype=bool)
    smooth_translations = translations.copy()
    for t in range(t_count):
        pelvis_src = pelvis_point(whole[t], scores_wb[t])
        pelvis_dst = np.asarray(core[t, 0], dtype=np.float64)
        if pelvis_src is not None and np.all(np.isfinite(pelvis_dst)):
            smooth_translations[t] = (
                pelvis_dst - float(smooth_scale[t]) * (smooth_rotation[t] @ pelvis_src)
            )
        src, dst, _ = shared_anchor_points(core[t], whole[t], scores_wb[t], score_thr)
        if src.shape[0] >= min_anchors:
            per_frame_residual[t] = alignment_residual(
                src,
                dst,
                smooth_rotation[t],
                float(smooth_scale[t]),
                smooth_translations[t],
            )
            accepted[t] = per_frame_residual[t] <= residual_thr
        else:
            accepted[t] = False
        aligned = apply_similarity(
            whole[t, :, :3],
            smooth_rotation[t],
            float(smooth_scale[t]),
            smooth_translations[t],
        )
        frame_valid = (
            np.all(np.isfinite(aligned), axis=-1)
            & np.isfinite(scores_wb[t])
            & (scores_wb[t] >= score_thr)
            & accepted[t]
        )
        whole_aligned[t, frame_valid] = aligned[frame_valid]
        whole_valid[t] = frame_valid
        for core_index, wb_index in (
            (0, None),
            (1, 12), (2, 14), (3, 16), (4, 11), (5, 13), (6, 15),
            (9, 0), (11, 5), (12, 7), (13, 9), (14, 6), (15, 8), (16, 10),
        ):
            if core_index == 0:
                pelvis = pelvis_point(aligned, scores_wb[t])
                aligned_core_from_wb[t, 0] = (
                    pelvis if pelvis is not None else aligned[[11, 12], :3].mean(axis=0)
                )
            else:
                aligned_core_from_wb[t, core_index] = aligned[wb_index]

    extras_raw, extras_scores, extras_valid = extract_extended_from_wholebody(whole, scores_wb, score_thr)
    extras_aligned = np.full_like(extras_raw, np.nan)
    for t in range(t_count):
        extras_aligned[t] = apply_similarity(
            extras_raw[t],
            smooth_rotation[t],
            float(smooth_scale[t]),
            smooth_translations[t],
        )
        extras_valid[t] &= accepted[t]
        extras_scores[t] = np.where(extras_valid[t], extras_scores[t], 0.0)

    length_medians: Dict[int, float] = {}
    for group in ATTACHMENT_GROUPS:
        parent = int(group["parent"])
        for index in group["indices"]:
            idx = int(index)
            lengths = []
            for t in range(t_count):
                if not extras_valid[t, idx]:
                    continue
                parent_pt = aligned_core_from_wb[t, parent]
                if not np.all(np.isfinite(parent_pt)):
                    continue
                lengths.append(float(np.linalg.norm(extras_aligned[t, idx] - parent_pt)))
            if lengths:
                length_medians[idx] = float(np.median(lengths))

    fused_extras = np.full_like(extras_aligned, np.nan)
    fused_valid = extras_valid.copy()
    for t in range(t_count):
        frame_fused, frame_valid = local_attach_frame(
            core[t],
            extras_aligned[t],
            extras_valid[t],
            aligned_core_from_wb[t],
            length_medians,
            length_clip,
        )
        fused_extras[t] = frame_fused
        fused_valid[t] = frame_valid

    interp_coords, interp_scores, interp_valid, repaired = interpolate_gaps(
        np.nan_to_num(fused_extras, nan=0.0),
        extras_scores,
        fused_valid,
        max_gap=max_gap,
        repaired_score=repaired_score,
    )
    fused_extras = interp_coords
    extras_scores = interp_scores
    fused_valid = interp_valid

    if euro.get("enabled", True):
        filtered = fused_extras.copy()
        for j in range(NUM_CORE_JOINTS, NUM_EXTENDED_JOINTS):
            filtered[:, j] = filter_sequence_one_euro(
                fused_extras[:, j],
                valid=fused_valid[:, j],
                fps=fps,
                min_cutoff=float(euro.get("min_cutoff", 1.2)),
                beta=float(euro.get("beta", 0.01)),
                dcutoff=float(euro.get("dcutoff", 1.0)),
            )
        fused_extras = filtered

    extended = np.full((t_count, NUM_EXTENDED_JOINTS, 3), np.nan, dtype=np.float64)
    extended[:, :NUM_CORE_JOINTS] = core[:, :, :3]
    extended[:, NUM_CORE_JOINTS:] = fused_extras[:, NUM_CORE_JOINTS:]

    out_scores = np.ones((t_count, NUM_EXTENDED_JOINTS), dtype=np.float64)
    out_scores[:, NUM_CORE_JOINTS:] = extras_scores[:, NUM_CORE_JOINTS:]
    out_valid = np.ones((t_count, NUM_EXTENDED_JOINTS), dtype=bool)
    out_valid[:, :NUM_CORE_JOINTS] = _valid_mask(core[:, :, :3])
    out_valid[:, NUM_CORE_JOINTS:] = fused_valid[:, NUM_CORE_JOINTS:]
    extended[:, NUM_CORE_JOINTS:][~out_valid[:, NUM_CORE_JOINTS:]] = np.nan

    source_status = np.full((t_count, NUM_EXTENDED_JOINTS), SOURCE_STATUS_HIDDEN, dtype=np.int32)
    source_status[:, :NUM_CORE_JOINTS] = SOURCE_STATUS_CORE
    extra_mask = out_valid[:, NUM_CORE_JOINTS:]
    source_status[:, NUM_CORE_JOINTS:][extra_mask] = SOURCE_STATUS_EXTENDED
    source_status[:, NUM_CORE_JOINTS:][repaired[:, NUM_CORE_JOINTS:] & extra_mask] = SOURCE_STATUS_INTERPOLATED
    source_status[:, NUM_CORE_JOINTS:][~accepted[:, None] & extras_valid[:, NUM_CORE_JOINTS:]] = SOURCE_STATUS_REJECTED

    stats = {
        "frames": int(t_count),
        "fps": fps,
        "schema_version": SCHEMA_VERSION,
        "accepted_frames": int(np.sum(accepted)),
        "rejected_frames": int(np.sum(~accepted)),
        "mean_residual": float(np.nanmean(per_frame_residual)) if np.any(np.isfinite(per_frame_residual)) else None,
        "valid_extended_mean": float(np.mean(out_valid[:, NUM_CORE_JOINTS:])),
        "interpolated_count": int(np.sum(repaired[:, NUM_CORE_JOINTS:])),
        "global": global_info,
        "smooth_scale_median": float(np.median(smooth_scale)),
    }
    return {
        "core_pose_3d": core[:, :, :3].astype(np.float32),
        "extended_pose_3d": extended.astype(np.float32),
        "extended_scores": out_scores.astype(np.float32),
        "extended_valid": out_valid,
        "alignment_residual": per_frame_residual.astype(np.float32),
        "alignment_accepted": accepted,
        "source_status": source_status,
        "rotation": rotation.astype(np.float32),
        "rotation_per_frame": smooth_rotation.astype(np.float32),
        "scale_per_frame": smooth_scale.astype(np.float32),
        "wholebody_pose_3d_aligned": whole_aligned.astype(np.float32),
        "wholebody_scores": scores_wb.astype(np.float32),
        "wholebody_valid": whole_valid,
        "stats": stats,
    }


def build_extended_npz_payload(source_3d: dict, fused: dict, extra_meta: Optional[dict] = None) -> dict:
    payload = {key: source_3d[key] for key in source_3d.files} if hasattr(source_3d, "files") else dict(source_3d)
    payload["pred3d_root_relative_image_units"] = fused["core_pose_3d"]
    payload["core_pose_3d"] = fused["core_pose_3d"]
    payload["extended_pose_3d"] = fused["extended_pose_3d"]
    payload["extended_scores"] = fused["extended_scores"]
    payload["extended_valid"] = fused["extended_valid"]
    payload["alignment_residual"] = fused["alignment_residual"]
    payload["alignment_accepted"] = fused["alignment_accepted"]
    payload["extended_source_status"] = fused["source_status"]
    payload["extended_joint_names"] = np.asarray(EXTENDED_JOINT_NAMES)
    payload["wholebody_pose_3d_aligned"] = fused["wholebody_pose_3d_aligned"]
    payload["wholebody_scores"] = fused["wholebody_scores"]
    payload["wholebody_valid"] = fused["wholebody_valid"]
    meta = schema_metadata()
    if extra_meta:
        meta.update(extra_meta)
    payload["extended_schema"] = np.asarray(json.dumps(meta, ensure_ascii=False))
    payload["extended_stats"] = np.asarray(json.dumps(fused["stats"], ensure_ascii=False))
    return payload
