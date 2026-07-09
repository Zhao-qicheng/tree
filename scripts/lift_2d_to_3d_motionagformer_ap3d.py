import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch import nn


H36M_NAMES = [
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
]

H36M_PAIRS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_motionagformer(motionagformer_root, config_path, checkpoint_path, device):
    motionagformer_root = Path(motionagformer_root)
    sys.path.insert(0, str(motionagformer_root))

    from model.MotionAGFormer import MotionAGFormer

    cfg = load_config(config_path)
    act_mapper = {"gelu": nn.GELU, "relu": nn.ReLU}
    model = MotionAGFormer(
        n_layers=cfg["n_layers"],
        dim_in=cfg["dim_in"],
        dim_feat=cfg["dim_feat"],
        dim_rep=cfg["dim_rep"],
        dim_out=cfg["dim_out"],
        mlp_ratio=cfg["mlp_ratio"],
        act_layer=act_mapper[cfg["act_layer"]],
        attn_drop=cfg["attn_drop"],
        drop=cfg["drop"],
        drop_path=cfg["drop_path"],
        use_layer_scale=cfg["use_layer_scale"],
        layer_scale_init_value=cfg["layer_scale_init_value"],
        use_adaptive_fusion=cfg["use_adaptive_fusion"],
        num_heads=cfg["num_heads"],
        qkv_bias=cfg["qkv_bias"],
        qkv_scale=cfg["qkv_scale"],
        hierarchical=cfg["hierarchical"],
        num_joints=cfg["num_joints"],
        use_temporal_similarity=cfg["use_temporal_similarity"],
        temporal_connection_len=cfg["temporal_connection_len"],
        use_tcn=cfg["use_tcn"],
        graph_only=cfg["graph_only"],
        neighbour_num=cfg["neighbour_num"],
        n_frames=cfg["n_frames"],
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. missing={missing[:5]}, unexpected={unexpected[:5]}")

    model.to(device)
    model.eval()
    return model, cfg, checkpoint


def normalize_ap3d_2d(h36m_keypoints, width, height):
    result = h36m_keypoints.astype(np.float32).copy()
    result[..., 0] = result[..., 0] / width * 2.0 - 1.0
    result[..., 1] = result[..., 1] / width * 2.0 - height / width
    result[..., 2] = np.clip(result[..., 2], 0.0, 1.0)
    return result


def make_windows(num_frames, window_size, stride):
    if num_frames <= 0:
        raise ValueError("No frames found in input 2D sequence.")
    if num_frames <= window_size:
        return [np.arange(window_size, dtype=np.int64) % num_frames]

    starts = list(range(0, num_frames - window_size + 1, stride))
    last_start = num_frames - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return [np.arange(start, start + window_size, dtype=np.int64) for start in starts]


def aggregate_predictions(predictions, windows, num_frames):
    output = np.zeros((num_frames, 17, 3), dtype=np.float32)
    counts = np.zeros((num_frames, 1, 1), dtype=np.float32)
    for pred, window in zip(predictions, windows):
        for local_idx, frame_idx in enumerate(window):
            output[frame_idx] += pred[local_idx]
            counts[frame_idx] += 1.0
    counts[counts == 0] = 1.0
    return output / counts


def draw_3d_video(pred3d, output_path, fps=29.0, size=(900, 700)):
    width, height = size
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Orthographic view for quick checking, not a calibrated camera projection.
    yaw = np.deg2rad(35.0)
    pitch = np.deg2rad(15.0)
    ry = np.array(
        [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]],
        dtype=np.float32,
    )
    rx = np.array(
        [[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]],
        dtype=np.float32,
    )
    rotation = rx @ ry

    valid_scale = np.percentile(np.abs(pred3d.reshape(-1, 3)), 95)
    scale = 260.0 / max(float(valid_scale), 1e-6)

    for pose in pred3d:
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        pts = pose.copy()
        pts = pts @ rotation.T
        pts2d = pts[:, :2] * scale + np.array([width * 0.5, height * 0.58], dtype=np.float32)

        for a, b in H36M_PAIRS:
            pa = tuple(np.round(pts2d[a]).astype(int))
            pb = tuple(np.round(pts2d[b]).astype(int))
            cv2.line(canvas, pa, pb, (30, 80, 230), 3)
        for point in pts2d:
            cv2.circle(canvas, tuple(np.round(point).astype(int)), 4, (20, 180, 60), -1)

        cv2.putText(
            canvas,
            "root-relative AP3D/MotionAGFormer 3D",
            (24, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )
        writer.write(canvas)

    writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-2d-npz", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--motionagformer-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--out-json")
    parser.add_argument("--vis-out")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, cfg, checkpoint = load_motionagformer(
        args.motionagformer_root, args.config, args.checkpoint, device
    )

    data = np.load(args.input_2d_npz, allow_pickle=True)
    h36m_2d = data["h36m_keypoints"].astype(np.float32)
    width = int(np.asarray(data["image_width"]).item())
    height = int(np.asarray(data["image_height"]).item())

    input_2d = normalize_ap3d_2d(h36m_2d, width, height)
    window_size = int(cfg["n_frames"])
    windows = make_windows(len(input_2d), window_size, args.stride)

    clip_predictions = []
    with torch.no_grad():
        for window in windows:
            clip = torch.from_numpy(input_2d[window][None]).to(device)
            pred = model(clip).cpu().numpy()[0]
            # AP3D/MotionAGFormer-small checkpoint is trained root-relative.
            pred[:, 0, :] = 0.0
            clip_predictions.append(pred.astype(np.float32))

    pred3d_norm = aggregate_predictions(clip_predictions, windows, len(input_2d))
    pred3d_root_relative_image_units = pred3d_norm * (width / 2.0)
    pred3d_with_2d_root = pred3d_root_relative_image_units.copy()
    pred3d_with_2d_root[:, :, :2] += h36m_2d[:, 0:1, :2]

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        pred3d_norm=pred3d_norm,
        pred3d_root_relative_image_units=pred3d_root_relative_image_units,
        pred3d_with_2d_root=pred3d_with_2d_root,
        input_2d_norm=input_2d,
        input_2d_pixels=h36m_2d,
        windows=np.asarray(windows, dtype=np.int32),
        image_width=np.asarray(width, dtype=np.int32),
        image_height=np.asarray(height, dtype=np.int32),
        h36m_joint_names=np.asarray(H36M_NAMES),
        source_2d_npz=str(args.input_2d_npz),
        checkpoint=str(args.checkpoint),
        checkpoint_epoch=np.asarray(checkpoint.get("epoch", -1), dtype=np.int32),
        checkpoint_min_mpjpe=np.asarray(checkpoint.get("min_mpjpe", np.nan), dtype=np.float32),
    )

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_2d_npz": str(args.input_2d_npz),
            "checkpoint": str(args.checkpoint),
            "format": "h36m_17_root_relative_3d",
            "units": "image_width_scaled_relative_units",
            "joint_names": H36M_NAMES,
            "frames": [
                {
                    "frame_id": i,
                    "keypoints_3d": pred3d_root_relative_image_units[i].tolist(),
                }
                for i in range(len(pred3d_root_relative_image_units))
            ],
        }
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    if args.vis_out:
        vis_out = Path(args.vis_out)
        vis_out.parent.mkdir(parents=True, exist_ok=True)
        draw_3d_video(pred3d_root_relative_image_units, vis_out)

    print(f"frames: {len(input_2d)}")
    print(f"windows: {len(windows)} x {window_size}")
    print(f"output shape: {pred3d_norm.shape}")
    print(f"saved: {out_npz}")
    if args.out_json:
        print(f"saved: {args.out_json}")
    if args.vis_out:
        print(f"saved: {args.vis_out}")


if __name__ == "__main__":
    main()
