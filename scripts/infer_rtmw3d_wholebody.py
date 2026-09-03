"""对视频运行 RTMW3D，输出与主运动员对齐的 WholeBody-133 原始结果。

默认不覆盖已有 17 点输出。身份来自 H36M 2D NPZ 中的 coco_keypoints，
而不是每帧重新挑选最高分人物。
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.extended_pose_schema import NUM_WHOLEBODY_JOINTS  # noqa: E402


DEFAULT_RTMW3D_WEIGHTS = (
    "https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/"
    "rtmw3d/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth"
)


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def bbox_from_keypoints(keypoints: np.ndarray, width: int, height: int, pad_ratio: float = 0.35, score_thr: float = 0.05):
    kpts = np.asarray(keypoints, dtype=np.float32)
    if kpts.ndim != 2 or kpts.shape[1] < 2:
        return None
    scores = kpts[:, 2] if kpts.shape[1] >= 3 else np.ones((kpts.shape[0],), dtype=np.float32)
    valid = np.isfinite(kpts[:, 0]) & np.isfinite(kpts[:, 1]) & (scores >= score_thr)
    if int(np.sum(valid)) < 4:
        return None
    xy = kpts[valid, :2]
    x1, y1 = np.min(xy, axis=0)
    x2, y2 = np.max(xy, axis=0)
    pad = float(pad_ratio) * max(float(x2 - x1), float(y2 - y1), 1.0)
    box = np.array(
        [
            max(0.0, x1 - pad),
            max(0.0, y1 - pad),
            min(float(width - 1), x2 + pad),
            min(float(height - 1), y2 + pad),
        ],
        dtype=np.float32,
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def bbox_iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_instance(instances, target_coco, target_bbox, image_diag: float):
    best_idx = -1
    best_score = -1e9
    for idx, inst in enumerate(instances):
        kpts2d = np.asarray(inst.get("keypoints_2d"), dtype=np.float32)
        scores = np.asarray(inst.get("keypoint_scores"), dtype=np.float32)
        bbox = inst.get("bbox")
        score = 0.0
        if target_coco is not None and kpts2d is not None and kpts2d.shape[0] >= 17:
            valid = (target_coco[:17, 2] > 0.05) & np.isfinite(kpts2d[:17, 0])
            if int(np.sum(valid)) >= 4:
                dist = float(np.mean(np.linalg.norm(kpts2d[:17][valid] - target_coco[:17, :2][valid], axis=-1)))
                score -= dist / max(image_diag, 1.0)
        score += 0.35 * bbox_iou(bbox, target_bbox)
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_idx < 0 or best_score < -0.25:
        return None
    return best_idx


def squeeze_kpts(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    while array.ndim > 2:
        array = array[0]
    return array


def parse_pred_instance(result):
    pred = result.pred_instances
    keypoints = squeeze_kpts(pred.keypoints)
    keypoints_2d = squeeze_kpts(getattr(pred, "transformed_keypoints", keypoints[..., :2]))
    scores = np.asarray(pred.keypoint_scores, dtype=np.float32)
    while scores.ndim > 1:
        scores = scores[0]
    bboxes = np.asarray(pred.bboxes, dtype=np.float32).reshape(-1, 4)
    bbox = bboxes[0] if len(bboxes) else np.zeros((4,), dtype=np.float32)
    if keypoints.shape[0] != NUM_WHOLEBODY_JOINTS:
        return None
    return {
        "keypoints_3d": keypoints[:, :3],
        "keypoints_2d": keypoints_2d[:, :2],
        "keypoint_scores": scores[:NUM_WHOLEBODY_JOINTS],
        "bbox": bbox[:4],
    }


def atomic_savez(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    tmp_path.replace(path)


def load_progress(path: Path, frame_count: int):
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=True)
    processed = np.asarray(data["processed"], dtype=bool)
    if processed.shape[0] != frame_count:
        log(f"Resume file frame count mismatch ({processed.shape[0]} vs {frame_count}), starting over")
        return None
    return {key: data[key] for key in data.files}


def resolve_scoped_config(config_path: Path, mmpose_root: Path) -> Path:
    """Resolve mmpose::_base_ references when running from an uninstalled source tree."""
    text = config_path.read_text(encoding="utf-8")
    marker = "mmpose::_base_/"
    if marker not in text:
        return config_path
    local_base = (mmpose_root / "configs" / "_base_").resolve().as_posix() + "/"
    resolved = text.replace(marker, local_base)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix="rtmw3d_resolved_",
        encoding="utf-8",
        delete=False,
    )
    with handle:
        handle.write(resolved)
    return Path(handle.name)


def main():
    parser = argparse.ArgumentParser(description="Run RTMW3D whole-body inference for the tracked main skater.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--out-json")
    parser.add_argument("--mmpose-root", required=True)
    parser.add_argument("--rtmw3d-config")
    parser.add_argument("--rtmw3d-weights", default=DEFAULT_RTMW3D_WEIGHTS)
    parser.add_argument("--det-model")
    parser.add_argument("--det-weights", required=True)
    parser.add_argument("--h36m-2d-npz", help="Existing H36M 2D NPZ used to reuse main-skater identity.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bbox-thr", type=float, default=0.35)
    parser.add_argument("--kpt-thr", type=float, default=0.25)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--det-cat-id", type=int, default=0)
    args = parser.parse_args()

    mmpose_root = Path(args.mmpose_root)
    rtmpose3d_root = mmpose_root / "projects" / "rtmpose3d"
    if not rtmpose3d_root.is_dir():
        raise SystemExit(f"RTMPose3D project not found: {rtmpose3d_root}")
    sys.path.insert(0, str(rtmpose3d_root))
    sys.path.insert(0, str(mmpose_root))

    from mmpose.apis import inference_topdown, init_model  # noqa: E402
    from mmpose.utils import adapt_mmdet_pipeline  # noqa: E402
    import rtmpose3d  # noqa: F401,E402

    try:
        from mmdet.apis import inference_detector, init_detector
    except ImportError as exc:
        raise SystemExit("mmdet is required for RTMW3D inference") from exc

    rtmw3d_config = (
        Path(args.rtmw3d_config)
        if args.rtmw3d_config
        else rtmpose3d_root / "configs" / "rtmw3d-l_8xb64_cocktail14-384x288.py"
    )
    det_model = (
        Path(args.det_model)
        if args.det_model
        else mmpose_root / "demo" / "mmdetection_cfg" / "rtmdet_m_640-8xb32_coco-person.py"
    )
    if not rtmw3d_config.is_file():
        raise SystemExit(f"RTMW3D config not found: {rtmw3d_config}")

    video_path = Path(args.video)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if args.max_frames and total:
        total = min(total, args.max_frames)
    elif args.max_frames:
        total = args.max_frames
    if total <= 0:
        # fallback: count by reading
        total = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            total += 1
            if args.max_frames and total >= args.max_frames:
                break
        cap.release()
        cap = cv2.VideoCapture(str(video_path))

    coco_kpts = None
    if args.h36m_2d_npz:
        two_d = np.load(args.h36m_2d_npz, allow_pickle=True)
        if "coco_keypoints" in two_d.files:
            coco_kpts = np.asarray(two_d["coco_keypoints"], dtype=np.float32)
            if coco_kpts.shape[0] < total:
                log(f"H36M 2D frames ({coco_kpts.shape[0]}) < video frames ({total}); extra frames use detector matching")

    out_npz = Path(args.out_npz)
    payload = None
    if args.resume:
        payload = load_progress(out_npz, total)
        if payload is not None:
            log(f"Resuming from {out_npz}, processed {int(np.sum(payload['processed']))}/{total}")

    if payload is None:
        payload = {
            "keypoints_3d_cam": np.full((total, NUM_WHOLEBODY_JOINTS, 3), np.nan, dtype=np.float32),
            "keypoints_2d": np.full((total, NUM_WHOLEBODY_JOINTS, 2), np.nan, dtype=np.float32),
            "keypoint_scores": np.zeros((total, NUM_WHOLEBODY_JOINTS), dtype=np.float32),
            "bboxes": np.zeros((total, 4), dtype=np.float32),
            "selected": np.zeros((total,), dtype=bool),
            "processed": np.zeros((total,), dtype=bool),
            "frame_ids": np.arange(total, dtype=np.int32),
            "image_width": np.asarray(width, dtype=np.int32),
            "image_height": np.asarray(height, dtype=np.int32),
            "source_video": str(video_path),
            "source_2d_npz": str(args.h36m_2d_npz or ""),
            "model_config": str(rtmw3d_config),
            "model_checkpoint": str(args.rtmw3d_weights),
        }

    log("Init detector and RTMW3D")
    detector = init_detector(str(det_model), args.det_weights, device=args.device.lower())
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    resolved_config = resolve_scoped_config(rtmw3d_config, mmpose_root)
    try:
        pose_estimator = init_model(
            str(resolved_config),
            args.rtmw3d_weights,
            device=args.device.lower(),
        )
    finally:
        if resolved_config != rtmw3d_config:
            resolved_config.unlink(missing_ok=True)
    pose_estimator.cfg.model.test_cfg.mode = "3d"

    image_diag = float(np.hypot(width, height))
    prev_bbox = None
    selected_count = 0
    for frame_idx in range(total):
        ok, frame = cap.read()
        if not ok:
            log(f"Video ended early at frame {frame_idx}")
            break
        if payload["processed"][frame_idx]:
            if payload["selected"][frame_idx]:
                prev_bbox = payload["bboxes"][frame_idx]
                selected_count += 1
            continue

        target_coco = coco_kpts[frame_idx] if coco_kpts is not None and frame_idx < len(coco_kpts) else None
        target_bbox = bbox_from_keypoints(target_coco, width, height) if target_coco is not None else None
        if target_bbox is None:
            target_bbox = prev_bbox

        chosen = None
        if target_bbox is not None:
            pose_results = inference_topdown(pose_estimator, frame, bboxes=target_bbox.reshape(1, 4))
            parsed = [parse_pred_instance(item) for item in pose_results]
            parsed = [item for item in parsed if item is not None]
            if parsed:
                chosen = parsed[0]
        if chosen is None:
            det_result = inference_detector(detector, frame)
            pred = det_result.pred_instances.cpu().numpy()
            keep = (pred.labels == args.det_cat_id) & (pred.scores > args.bbox_thr)
            det_bboxes = pred.bboxes[keep]
            if len(det_bboxes) == 0:
                payload["processed"][frame_idx] = True
                continue
            pose_results = inference_topdown(pose_estimator, frame, bboxes=det_bboxes)
            parsed = [parse_pred_instance(item) for item in pose_results]
            parsed = [item for item in parsed if item is not None]
            match_idx = match_instance(parsed, target_coco, target_bbox, image_diag)
            if match_idx is not None:
                chosen = parsed[match_idx]

        payload["processed"][frame_idx] = True
        if chosen is None:
            continue
        payload["keypoints_3d_cam"][frame_idx] = chosen["keypoints_3d"]
        payload["keypoints_2d"][frame_idx] = chosen["keypoints_2d"]
        payload["keypoint_scores"][frame_idx] = chosen["keypoint_scores"]
        payload["bboxes"][frame_idx] = chosen["bbox"]
        payload["selected"][frame_idx] = True
        prev_bbox = chosen["bbox"]
        selected_count += 1

        if args.save_every and frame_idx > 0 and frame_idx % args.save_every == 0:
            atomic_savez(out_npz, **payload)
            log(f"Checkpoint frame {frame_idx}/{total}, selected={selected_count}")

    cap.release()
    atomic_savez(out_npz, **payload)
    log(f"Saved {out_npz}")
    print(f"frames: {total}")
    print(f"selected frames: {int(np.sum(payload['selected']))}")
    print(f"saved: {out_npz}")

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "source_video": str(video_path),
            "source_2d_npz": str(args.h36m_2d_npz or ""),
            "format": "coco_wholebody_133_rtmw3d",
            "frames": int(total),
            "selected_frames": int(np.sum(payload["selected"])),
            "model_config": str(rtmw3d_config),
            "model_checkpoint": str(args.rtmw3d_weights),
        }
        with out_json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        print(f"saved: {out_json}")


if __name__ == "__main__":
    main()
