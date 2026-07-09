import argparse
import json
from pathlib import Path

import cv2
import numpy as np


COCO_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# MotionAGFormer demo mapping. Keep this order for its pretrained H36M-style models.
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

H36M_COCO_ORDER = [9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
COCO_ORDER = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
SYNTHETIC_KEYPOINTS = [10, 8, 0, 7]  # head, thorax, pelvis/root, spine
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


def bbox_area(bbox):
    if bbox is None:
        return 0.0
    box = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if box.size < 4:
        return 0.0
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def bbox_center(bbox):
    box = np.asarray(bbox, dtype=np.float32).reshape(-1)
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32)


def bbox_iou(a, b):
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def instance_arrays(instance):
    keypoints = np.asarray(instance.get("keypoints", []), dtype=np.float32)
    scores = np.asarray(instance.get("keypoint_scores", []), dtype=np.float32)
    if keypoints.shape != (17, 2) or scores.shape != (17,):
        return None, None
    coco = np.concatenate([keypoints, scores[:, None]], axis=1)
    return coco, scores


def select_instances(frames, image_width, image_height):
    diag = float(np.hypot(image_width, image_height)) or 1.0
    selected = []
    prev_bbox = None
    prev_center = None

    for frame in frames:
        instances = frame.get("instances", [])
        candidates = []
        for idx, inst in enumerate(instances):
            coco, scores = instance_arrays(inst)
            if coco is None:
                continue
            bbox = np.asarray(inst.get("bbox", []), dtype=np.float32).reshape(-1)
            if bbox.size < 4:
                continue
            area = bbox_area(bbox)
            if area <= 1:
                continue
            bbox_score = float(np.asarray(inst.get("bbox_score", 0.0)).reshape(-1)[0])
            kp_score = float(np.nanmean(scores))
            candidates.append(
                {
                    "idx": idx,
                    "instance": inst,
                    "bbox": bbox[:4],
                    "center": bbox_center(bbox[:4]),
                    "area": area,
                    "bbox_score": bbox_score,
                    "kp_score": kp_score,
                }
            )

        if not candidates:
            selected.append(None)
            continue

        max_area = max(c["area"] for c in candidates) or 1.0
        if prev_center is None:
            best = max(
                candidates,
                key=lambda c: (c["area"] / max_area) + 0.3 * c["bbox_score"] + 0.2 * c["kp_score"],
            )
        else:
            def track_score(c):
                dist = float(np.linalg.norm(c["center"] - prev_center)) / diag
                return (
                    0.55 * bbox_iou(c["bbox"], prev_bbox)
                    - 0.75 * dist
                    + 0.25 * (c["area"] / max_area)
                    + 0.2 * c["bbox_score"]
                    + 0.15 * c["kp_score"]
                )

            best = max(candidates, key=track_score)

        selected.append(best)
        prev_bbox = best["bbox"]
        prev_center = best["center"]

    return selected


def coco_to_h36m(coco):
    coords = coco[:, :, :2]
    scores = coco[:, :, 2]
    temporal = coco.shape[0]

    h36m_xy = np.zeros((temporal, 17, 2), dtype=np.float32)
    h36m_score = np.zeros((temporal, 17), dtype=np.float32)

    synthetic_xy = np.zeros((temporal, 4, 2), dtype=np.float32)
    synthetic_score = np.zeros((temporal, 4), dtype=np.float32)

    # head, thorax, pelvis/root, spine. This mirrors MotionAGFormer demo logic.
    synthetic_xy[:, 0, 0] = np.mean(coords[:, 1:5, 0], axis=1, dtype=np.float32)
    synthetic_xy[:, 0, 1] = np.sum(coords[:, 1:3, 1], axis=1, dtype=np.float32) - coords[:, 0, 1]
    synthetic_score[:, 0] = np.mean(scores[:, [0, 1, 2, 3, 4]], axis=1, dtype=np.float32)

    synthetic_xy[:, 1, :] = np.mean(coords[:, 5:7, :], axis=1, dtype=np.float32)
    synthetic_xy[:, 1, :] += (coords[:, 0, :] - synthetic_xy[:, 1, :]) / 3
    synthetic_score[:, 1] = np.mean(scores[:, [0, 5, 6]], axis=1, dtype=np.float32)

    synthetic_xy[:, 2, :] = np.mean(coords[:, 11:13, :], axis=1, dtype=np.float32)
    synthetic_score[:, 2] = np.mean(scores[:, [11, 12]], axis=1, dtype=np.float32)

    synthetic_xy[:, 3, :] = np.mean(coords[:, [5, 6, 11, 12], :], axis=1, dtype=np.float32)
    synthetic_score[:, 3] = np.mean(scores[:, [5, 6, 11, 12]], axis=1, dtype=np.float32)

    h36m_xy[:, SYNTHETIC_KEYPOINTS, :] = synthetic_xy
    h36m_score[:, SYNTHETIC_KEYPOINTS] = synthetic_score
    h36m_xy[:, H36M_COCO_ORDER, :] = coords[:, COCO_ORDER, :]
    h36m_score[:, H36M_COCO_ORDER] = scores[:, COCO_ORDER]

    h36m_xy[:, 9, :] -= (h36m_xy[:, 9, :] - np.mean(coords[:, 5:7, :], axis=1, dtype=np.float32)) / 4
    h36m_xy[:, 7, 0] += 0.3 * (
        h36m_xy[:, 7, 0] - np.mean(h36m_xy[:, [0, 8], 0], axis=1, dtype=np.float32)
    )
    h36m_xy[:, 8, 1] -= (
        np.mean(coords[:, 1:3, 1], axis=1, dtype=np.float32) - coords[:, 0, 1]
    ) * 2 / 3

    return np.concatenate([h36m_xy, h36m_score[:, :, None]], axis=2)


def draw_h36m_video(video_path, h36m, output_path, threshold=0.2):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 29
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    while frame_idx < len(h36m):
        ok, frame = cap.read()
        if not ok:
            break
        pts = h36m[frame_idx, :, :2]
        scores = h36m[frame_idx, :, 2]
        for a, b in H36M_PAIRS:
            if scores[a] >= threshold and scores[b] >= threshold:
                pa = tuple(np.round(pts[a]).astype(int))
                pb = tuple(np.round(pts[b]).astype(int))
                cv2.line(frame, pa, pb, (0, 220, 255), 3)
        for point, score in zip(pts, scores):
            if score >= threshold:
                cv2.circle(frame, tuple(np.round(point).astype(int)), 4, (0, 255, 0), -1)
        writer.write(frame)
        frame_idx += 1

    writer.release()
    cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-json", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--out-json")
    parser.add_argument("--vis-out")
    args = parser.parse_args()

    pred_json = Path(args.pred_json)
    video_path = Path(args.video)
    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    with pred_json.open("r", encoding="utf-8") as f:
        frames = json.load(f)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    selected = select_instances(frames, width, height)
    coco = np.zeros((len(frames), 17, 3), dtype=np.float32)
    selected_indices = np.full((len(frames),), -1, dtype=np.int32)

    for i, item in enumerate(selected):
        if item is None:
            continue
        selected_indices[i] = item["idx"]
        coco_i, _ = instance_arrays(item["instance"])
        coco[i] = coco_i

    h36m = coco_to_h36m(coco)
    frame_ids = np.asarray([frame.get("frame_id", i) for i, frame in enumerate(frames)], dtype=np.int32)

    np.savez_compressed(
        out_npz,
        coco_keypoints=coco,
        h36m_keypoints=h36m,
        selected_indices=selected_indices,
        frame_ids=frame_ids,
        image_width=np.asarray(width, dtype=np.int32),
        image_height=np.asarray(height, dtype=np.int32),
        coco_joint_names=np.asarray(COCO_NAMES),
        h36m_joint_names=np.asarray(H36M_NAMES),
        source_json=str(pred_json),
        source_video=str(video_path),
    )

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_json": str(pred_json),
            "source_video": str(video_path),
            "format": "h36m_17",
            "joint_names": H36M_NAMES,
            "frames": [
                {
                    "frame_id": int(frame_ids[i]),
                    "selected_instance": int(selected_indices[i]),
                    "keypoints": h36m[i].tolist(),
                }
                for i in range(len(frames))
            ],
        }
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    if args.vis_out:
        vis_out = Path(args.vis_out)
        vis_out.parent.mkdir(parents=True, exist_ok=True)
        draw_h36m_video(video_path, h36m, vis_out)

    valid = int(np.sum(selected_indices >= 0))
    print(f"frames: {len(frames)}")
    print(f"selected frames: {valid}")
    print(f"saved: {out_npz}")
    if args.out_json:
        print(f"saved: {args.out_json}")
    if args.vis_out:
        print(f"saved: {args.vis_out}")


if __name__ == "__main__":
    main()
