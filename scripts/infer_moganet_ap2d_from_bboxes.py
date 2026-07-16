import argparse
import contextlib
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOGANET_ROOT = Path(r"C:\Users\86158\Desktop\MogaNet-main")
DEFAULT_CHECKPOINT = ROOT / "test" / "moganet_b_ap2d_384x288.pth"

COCO_PAIRS = [
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
]


def log(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def format_seconds(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {seconds:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"


def get_3rd_point(a, b):
    direction = a - b
    return b + np.array([-direction[1], direction[0]], dtype=np.float32)


def rotate_point(point, angle_rad):
    sin_a = np.sin(angle_rad)
    cos_a = np.cos(angle_rad)
    return np.array(
        [point[0] * cos_a - point[1] * sin_a, point[0] * sin_a + point[1] * cos_a],
        dtype=np.float32,
    )


def get_affine_transform(center, scale, output_size, inverse=False):
    dst_w, dst_h = output_size
    src_w = scale[0]
    rot_rad = 0.0
    src_dir = rotate_point(np.array([0.0, src_w * -0.5], dtype=np.float32), rot_rad)
    dst_dir = np.array([0.0, dst_w * -0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    src[2, :] = get_3rd_point(src[0, :], src[1, :])

    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = dst[0, :] + dst_dir
    dst[2, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inverse:
        return cv2.getAffineTransform(dst, src)
    return cv2.getAffineTransform(src, dst)


def bbox_to_center_scale(bbox, image_size, padding=1.25):
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    center = np.array([x1 + width * 0.5, y1 + height * 0.5], dtype=np.float32)

    aspect_ratio = image_size[0] / image_size[1]
    if width > aspect_ratio * height:
        height = width / aspect_ratio
    elif width < aspect_ratio * height:
        width = height * aspect_ratio

    scale = np.array([width * padding, height * padding], dtype=np.float32)
    return center, scale


def transform_preds(coords, center, scale, heatmap_size):
    heatmap_w, heatmap_h = heatmap_size
    target = coords.copy()
    target[:, 0] = target[:, 0] * scale[0] / heatmap_w + center[0] - scale[0] * 0.5
    target[:, 1] = target[:, 1] * scale[1] / heatmap_h + center[1] - scale[1] * 0.5
    return target


def decode_heatmaps(heatmaps, centers, scales):
    batch, joints, heatmap_h, heatmap_w = heatmaps.shape
    flat = heatmaps.reshape(batch, joints, -1)
    idx = np.argmax(flat, axis=2)
    maxvals = np.max(flat, axis=2)

    coords = np.zeros((batch, joints, 2), dtype=np.float32)
    coords[..., 0] = idx % heatmap_w
    coords[..., 1] = idx // heatmap_w

    for b in range(batch):
        for j in range(joints):
            px = int(coords[b, j, 0])
            py = int(coords[b, j, 1])
            if 1 < px < heatmap_w - 1 and 1 < py < heatmap_h - 1:
                diff = np.array(
                    [
                        heatmaps[b, j, py, px + 1] - heatmaps[b, j, py, px - 1],
                        heatmaps[b, j, py + 1, px] - heatmaps[b, j, py - 1, px],
                    ],
                    dtype=np.float32,
                )
                coords[b, j] += np.sign(diff) * 0.25

    decoded = np.zeros_like(coords)
    for b in range(batch):
        decoded[b] = transform_preds(coords[b], centers[b], scales[b], (heatmap_w, heatmap_h))

    scores = np.clip(maxvals, 0.0, 1.0).astype(np.float32)
    return decoded, scores


class TopdownHeatmapSimpleHead(nn.Module):
    def __init__(self, in_channels=512, out_channels=17):
        super().__init__()
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.final_layer = nn.Conv2d(256, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.deconv_layers(x)
        return self.final_layer(x)


class MogaNetAP2D(nn.Module):
    def __init__(self, moganet_root):
        super().__init__()
        sys.path.insert(0, str(moganet_root))
        with contextlib.redirect_stdout(io.StringIO()):
            from models.moganet import MogaNet

        self.backbone = MogaNet(
            arch="base",
            init_value=1e-5,
            drop_path_rate=0.4,
            fork_feat=True,
        )
        self.keypoint_head = TopdownHeatmapSimpleHead(in_channels=512, out_channels=17)

    def forward(self, x):
        features = self.backbone(x)
        return self.keypoint_head(features[3])


def load_model(moganet_root, checkpoint_path, device):
    model = MogaNetAP2D(moganet_root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. missing={missing[:8]}, unexpected={unexpected[:8]}")
    model.to(device)
    model.eval()
    return model, checkpoint


def load_bbox_frames(path):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "frames" in data:
        return data["frames"]
    raise ValueError(f"Unsupported bbox JSON structure: {path}")


def extract_bboxes(frame_record, bbox_thr):
    bboxes = []
    source_indices = []
    for idx, instance in enumerate(frame_record.get("instances", [])):
        raw_bbox = np.asarray(instance.get("bbox", []), dtype=np.float32).reshape(-1)
        if raw_bbox.size < 4:
            continue
        bbox_score = float(np.asarray(instance.get("bbox_score", raw_bbox[4] if raw_bbox.size >= 5 else 1.0)).reshape(-1)[0])
        if bbox_score < bbox_thr:
            continue
        bbox = raw_bbox[:4].astype(np.float32)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        bboxes.append(bbox)
        source_indices.append(idx)
    return bboxes, source_indices


def crop_instances(frame, bboxes, input_size):
    tensors = []
    centers = []
    scales = []
    for bbox in bboxes:
        center, scale = bbox_to_center_scale(bbox, input_size)
        trans = get_affine_transform(center, scale, input_size)
        crop = cv2.warpAffine(frame, trans, input_size, flags=cv2.INTER_LINEAR)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        crop = (crop - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )
        tensors.append(crop.transpose(2, 0, 1))
        centers.append(center)
        scales.append(scale)
    return (
        np.asarray(tensors, dtype=np.float32),
        np.asarray(centers, dtype=np.float32),
        np.asarray(scales, dtype=np.float32),
    )


def infer_frame(model, frame, bboxes, input_size, batch_size, device):
    if not bboxes:
        return np.zeros((0, 17, 2), dtype=np.float32), np.zeros((0, 17), dtype=np.float32)

    crops, centers, scales = crop_instances(frame, bboxes, input_size)
    all_keypoints = []
    all_scores = []
    with torch.no_grad():
        for start in range(0, len(crops), batch_size):
            end = start + batch_size
            tensor = torch.from_numpy(crops[start:end]).to(device)
            heatmaps = model(tensor).detach().cpu().numpy()
            keypoints, scores = decode_heatmaps(heatmaps, centers[start:end], scales[start:end])
            all_keypoints.append(keypoints)
            all_scores.append(scores)

    return np.concatenate(all_keypoints, axis=0), np.concatenate(all_scores, axis=0)


def draw_coco(frame, instances, threshold):
    for instance in instances:
        pts = np.asarray(instance["keypoints"], dtype=np.float32)
        scores = np.asarray(instance["keypoint_scores"], dtype=np.float32)
        for a, b in COCO_PAIRS:
            if scores[a] >= threshold and scores[b] >= threshold:
                cv2.line(
                    frame,
                    tuple(np.round(pts[a]).astype(int)),
                    tuple(np.round(pts[b]).astype(int)),
                    (0, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
        for point, score in zip(pts, scores):
            if score >= threshold:
                cv2.circle(frame, tuple(np.round(point).astype(int)), 3, (0, 255, 0), -1, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Run standalone MogaNet AP2D 2D pose inference using bboxes from an existing MMPose JSON."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--bbox-json", required=True, help="MMPose inferencer JSON used only as the person bbox source.")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--vis-out")
    parser.add_argument("--moganet-root", default=str(DEFAULT_MOGANET_ROOT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bbox-thr", type=float, default=0.2)
    parser.add_argument("--kpt-thr", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit-frames", type=int, default=0)
    args = parser.parse_args()

    video_path = Path(args.video)
    bbox_json = Path(args.bbox_json)
    out_json = Path(args.out_json)
    vis_out = Path(args.vis_out) if args.vis_out else None
    moganet_root = Path(args.moganet_root)
    checkpoint = Path(args.checkpoint)

    if not video_path.is_file():
        raise SystemExit(f"Input video not found: {video_path}")
    if not bbox_json.is_file():
        raise SystemExit(f"BBox JSON not found: {bbox_json}")
    if not moganet_root.is_dir():
        raise SystemExit(f"MogaNet root not found: {moganet_root}")
    if not checkpoint.is_file():
        raise SystemExit(f"MogaNet checkpoint not found: {checkpoint}")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    input_size = (288, 384)

    log(f"Loading MogaNet AP2D checkpoint: {checkpoint}")
    start = perf_counter()
    model, checkpoint_data = load_model(moganet_root, checkpoint, device)
    log(f"Model loaded after {format_seconds(perf_counter() - start)}")

    bbox_frames = load_bbox_frames(bbox_json)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if vis_out:
        vis_out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(vis_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    out_frames = []
    frame_idx = 0
    inference_start = perf_counter()
    while True:
        if args.limit_frames and frame_idx >= args.limit_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break

        bbox_record = bbox_frames[frame_idx] if frame_idx < len(bbox_frames) else {"frame_id": frame_idx, "instances": []}
        bboxes, source_indices = extract_bboxes(bbox_record, args.bbox_thr)
        keypoints, scores = infer_frame(model, frame, bboxes, input_size, max(1, args.batch_size), device)

        instances = []
        for i, bbox in enumerate(bboxes):
            instances.append(
                {
                    "bbox": [bbox.astype(float).tolist()],
                    "bbox_score": float(
                        np.asarray(
                            bbox_record.get("instances", [{}])[source_indices[i]].get("bbox_score", 1.0)
                        ).reshape(-1)[0]
                    ),
                    "bbox_source_instance": int(source_indices[i]),
                    "keypoints": keypoints[i].astype(float).tolist(),
                    "keypoint_scores": scores[i].astype(float).tolist(),
                }
            )

        out_frames.append({"frame_id": int(bbox_record.get("frame_id", frame_idx)), "instances": instances})

        if writer is not None:
            writer.write(draw_coco(frame, instances, args.kpt_thr))

        frame_idx += 1
        if frame_idx == 1 or frame_idx % 50 == 0:
            log(f"MogaNet AP2D processed {frame_idx} frames")

    cap.release()
    if writer is not None:
        writer.release()

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(out_frames, f, ensure_ascii=False)

    log(f"Saved MogaNet AP2D JSON: {out_json}")
    if vis_out:
        log(f"Saved MogaNet AP2D video: {vis_out}")
    log(f"Frames: {len(out_frames)}")
    log(f"Total MogaNet inference time: {format_seconds(perf_counter() - inference_start)}")
    if isinstance(checkpoint_data, dict) and "meta" in checkpoint_data:
        log("Checkpoint meta loaded")


if __name__ == "__main__":
    main()
