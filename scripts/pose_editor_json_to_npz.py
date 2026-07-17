import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_main_skater_h36m import H36M_NAMES, draw_h36m_video  # noqa: E402


def read_video_size(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def load_corrected_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames", [])
    if not frames:
        raise SystemExit(f"No frames found in corrected JSON: {path}")

    keypoints = []
    frame_ids = []
    selected = []
    for index, frame in enumerate(frames):
        pose = np.asarray(frame.get("keypoints", []), dtype=np.float32)
        if pose.shape != (17, 3):
            raise SystemExit(f"Invalid keypoint shape at frame index {index}: {pose.shape}")
        keypoints.append(pose)
        frame_ids.append(int(frame.get("frame_id", index)))
        selected.append(int(frame.get("selected_instance", -1)))

    return data, np.stack(keypoints, axis=0), np.asarray(frame_ids, dtype=np.int32), np.asarray(selected, dtype=np.int32)


def load_source_npz(path):
    if not path:
        return {}
    source = Path(path)
    if not source.is_file():
        raise SystemExit(f"Source NPZ not found: {source}")
    data = np.load(source, allow_pickle=True)
    return {key: data[key] for key in data.files}


def compatible_first_dim(value, frames):
    if not hasattr(value, "shape") or len(value.shape) == 0:
        return True
    return value.shape[0] == frames


def write_canonical_json(out_json, corrected_data, h36m, frame_ids, selected_indices, width, height):
    payload = {
        "source_json": corrected_data.get("source_json", ""),
        "source_npz": corrected_data.get("source_npz", ""),
        "source_video": corrected_data.get("source_video", ""),
        "correction_format": corrected_data.get("correction_format", "h36m_17_manual_2d_v1"),
        "format": "h36m_17",
        "image_width": int(width),
        "image_height": int(height),
        "fps": corrected_data.get("fps", None),
        "joint_names": corrected_data.get("joint_names", H36M_NAMES),
        "manual_edits": corrected_data.get("manual_edits", []),
        "frames": [
            {
                "frame_id": int(frame_ids[i]),
                "selected_instance": int(selected_indices[i]),
                "keypoints": h36m[i].tolist(),
            }
            for i in range(len(h36m))
        ],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrected-json", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--source-npz", default="")
    parser.add_argument("--video", default="")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--vis-out", default="")
    args = parser.parse_args()

    corrected_json = Path(args.corrected_json)
    corrected_data, h36m, frame_ids, selected_indices = load_corrected_json(corrected_json)
    frames = len(h36m)

    source_npz = args.source_npz or corrected_data.get("source_npz", "")
    source_payload = load_source_npz(source_npz)

    width = corrected_data.get("image_width")
    height = corrected_data.get("image_height")
    if width is None and "image_width" in source_payload:
        width = int(np.asarray(source_payload["image_width"]).item())
    if height is None and "image_height" in source_payload:
        height = int(np.asarray(source_payload["image_height"]).item())
    if (width is None or height is None) and args.video:
        width, height = read_video_size(Path(args.video))
    if width is None or height is None:
        raise SystemExit("image_width/image_height missing. Pass --source-npz or --video.")

    output_payload = {}
    for key, value in source_payload.items():
        if compatible_first_dim(value, frames):
            output_payload[key] = value

    output_payload["h36m_keypoints"] = h36m.astype(np.float32)
    output_payload["frame_ids"] = frame_ids
    output_payload["selected_indices"] = selected_indices
    output_payload["image_width"] = np.asarray(width, dtype=np.int32)
    output_payload["image_height"] = np.asarray(height, dtype=np.int32)
    output_payload["h36m_joint_names"] = np.asarray(corrected_data.get("joint_names", H36M_NAMES))
    output_payload["source_json"] = str(corrected_json)
    output_payload["source_video"] = corrected_data.get("source_video", args.video)
    output_payload["manual_edits"] = np.asarray(json.dumps(corrected_data.get("manual_edits", []), ensure_ascii=False))

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **output_payload)

    if args.out_json:
        write_canonical_json(Path(args.out_json), corrected_data, h36m, frame_ids, selected_indices, width, height)

    if args.vis_out:
        video_path = Path(args.video or corrected_data.get("source_video", ""))
        if not video_path.is_file():
            raise SystemExit(f"Video required for --vis-out, not found: {video_path}")
        vis_out = Path(args.vis_out)
        vis_out.parent.mkdir(parents=True, exist_ok=True)
        draw_h36m_video(video_path, h36m, vis_out)

    print(f"frames: {frames}")
    print(f"saved: {out_npz}")
    if args.out_json:
        print(f"saved: {args.out_json}")
    if args.vis_out:
        print(f"saved: {args.vis_out}")


if __name__ == "__main__":
    main()
