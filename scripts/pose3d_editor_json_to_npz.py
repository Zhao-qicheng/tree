import argparse
import json
from pathlib import Path

import cv2
import numpy as np


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


def load_source_npz(path):
    source = Path(path)
    if not source.is_file():
        raise SystemExit(f"Source NPZ not found: {source}")
    data = np.load(source, allow_pickle=True)
    return {key: data[key] for key in data.files}


def load_corrected_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames", [])
    if not frames:
        raise SystemExit(f"No frames found in corrected JSON: {path}")

    keypoints = []
    frame_ids = []
    for index, frame in enumerate(frames):
        pose = frame.get("keypoints_3d_display") or frame.get("keypoints_3d") or frame.get("keypoints")
        pose = np.asarray(pose, dtype=np.float32)
        if pose.shape != (17, 3):
            raise SystemExit(f"Invalid keypoint shape at frame index {index}: {pose.shape}")
        keypoints.append(pose)
        frame_ids.append(int(frame.get("frame_id", index)))
    return data, np.stack(keypoints, axis=0), np.asarray(frame_ids, dtype=np.int32)


def display_to_source(display):
    source = np.zeros_like(display, dtype=np.float32)
    source[..., 0] = display[..., 0]
    source[..., 1] = -display[..., 2]
    source[..., 2] = display[..., 1]
    return source


def draw_3d_video(pred3d, output_path, fps=29.0, size=(900, 700)):
    width, height = size
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

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
        pts = pose.copy() @ rotation.T
        pts2d = pts[:, :2] * scale + np.array([width * 0.5, height * 0.58], dtype=np.float32)
        for a, b in H36M_PAIRS:
            pa = tuple(np.round(pts2d[a]).astype(int))
            pb = tuple(np.round(pts2d[b]).astype(int))
            cv2.line(canvas, pa, pb, (30, 80, 230), 3)
        for point in pts2d:
            cv2.circle(canvas, tuple(np.round(point).astype(int)), 4, (20, 180, 60), -1)
        cv2.putText(
            canvas,
            "root-relative corrected 3D",
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
    parser.add_argument("--corrected-json", required=True)
    parser.add_argument("--source-npz", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--out-json")
    parser.add_argument("--vis-out")
    parser.add_argument("--video")
    args = parser.parse_args()

    corrected_json = Path(args.corrected_json)
    source_payload = load_source_npz(args.source_npz)
    corrected_data, display, frame_ids = load_corrected_json(corrected_json)
    source = display_to_source(display)
    frames = len(display)

    if "image_width" in source_payload:
        image_width = int(np.asarray(source_payload["image_width"]).item())
    elif corrected_data.get("image_width"):
        image_width = int(corrected_data["image_width"])
    else:
        image_width = 0
    if "image_height" in source_payload:
        image_height = int(np.asarray(source_payload["image_height"]).item())
    elif corrected_data.get("image_height"):
        image_height = int(corrected_data["image_height"])
    else:
        image_height = 0

    if image_width <= 0 or image_height <= 0:
        if args.video:
            cap = cv2.VideoCapture(str(Path(args.video)))
            if not cap.isOpened():
                raise SystemExit(f"Could not open video: {args.video}")
            image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        else:
            raise SystemExit("image_width/image_height missing. Pass --source-npz with metadata or --video.")

    output_payload = {}
    for key, value in source_payload.items():
        if hasattr(value, "shape") and value.shape and value.shape[0] == frames:
            output_payload[key] = value

    output_payload["pred3d_norm"] = source / (image_width / 2.0)
    output_payload["pred3d_root_relative_image_units"] = source.astype(np.float32)
    if "input_2d_pixels" in source_payload:
        input_2d_pixels = np.asarray(source_payload["input_2d_pixels"], dtype=np.float32)
        if input_2d_pixels.shape[0] == frames:
            pred3d_with_2d_root = source.copy()
            pred3d_with_2d_root[:, :, :2] += input_2d_pixels[:, 0:1, :2]
            output_payload["pred3d_with_2d_root"] = pred3d_with_2d_root
    output_payload["input_2d_pixels"] = source_payload.get("input_2d_pixels", np.zeros_like(source))
    if "input_2d_norm" in source_payload and np.asarray(source_payload["input_2d_norm"]).shape[0] == frames:
        output_payload["input_2d_norm"] = source_payload["input_2d_norm"]
    output_payload["windows"] = source_payload.get("windows", np.zeros((0, 0), dtype=np.int32))
    output_payload["image_width"] = np.asarray(image_width, dtype=np.int32)
    output_payload["image_height"] = np.asarray(image_height, dtype=np.int32)
    output_payload["h36m_joint_names"] = np.asarray(corrected_data.get("joint_names", H36M_NAMES))
    output_payload["source_3d_json"] = str(corrected_json)
    output_payload["source_3d_npz"] = str(Path(args.source_npz))
    output_payload["manual_3d_edits"] = np.asarray(json.dumps(corrected_data.get("manual_edits", []), ensure_ascii=False))

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **output_payload)

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_3d_json": str(corrected_json),
            "source_3d_npz": str(Path(args.source_npz)),
            "format": "h36m_17_corrected_3d",
            "coordinate_space": "source_x_y_z",
            "units": "image_width_scaled_relative_units",
            "joint_names": corrected_data.get("joint_names", H36M_NAMES),
            "frames": [
                {
                    "frame_id": int(frame_ids[i]),
                    "keypoints_3d": source[i].tolist(),
                }
                for i in range(frames)
            ],
        }
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    if args.vis_out:
        vis_out = Path(args.vis_out)
        vis_out.parent.mkdir(parents=True, exist_ok=True)
        draw_3d_video(source, vis_out)

    print(f"frames: {frames}")
    print(f"saved: {out_npz}")
    if args.out_json:
        print(f"saved: {args.out_json}")
    if args.vis_out:
        print(f"saved: {args.vis_out}")


if __name__ == "__main__":
    main()
