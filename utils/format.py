import os
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
from argparse import ArgumentParser


def get_marker_data(json_file: str) -> dict:
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)['Markers']


def get_main_range(parts: List[dict]) -> Tuple[int, int, dict]:
    ranges = [(part['Range']['Start'], part['Range']['End']) for part in parts]
    time_ranges = [end - start for start, end in ranges]

    main_idx = np.argmax(time_ranges)
    main_start, main_end = ranges[main_idx]

    # Adjust to be zero-indexed for array slicing
    return main_start - 1, main_end - 1, parts[main_idx]


def get_time_range(marker_data: List[dict]) -> Tuple[int, int]:
    # Initialize using the first marker's time range
    start, end, _ = get_main_range(marker_data[0]['Parts'])

    for marker in marker_data:
        tmp_start, tmp_end, _ = get_main_range(marker['Parts'])
        start = max(start, tmp_start)
        end = min(end, tmp_end)

    return start, end


def get_pose_array(marker_data: List[dict], time_range: Tuple[int, int]) -> Tuple[np.ndarray, List[str]]:
    start_time, end_time = time_range
    if start_time >= end_time or start_time < 0 or end_time < 0:
        raise ValueError(f"Invalid time range provided: {time_range}")

    labels = []
    pose3d = []

    for marker in marker_data:
        parts = marker['Parts']
        marker_start, _, main_part = get_main_range(parts)

        start = start_time - marker_start
        end = end_time - marker_start

        cood_3d = np.array(main_part['Values'])[start:end, :3]

        labels.append(marker['Name'])
        pose3d.append(cood_3d)

    pose3d = np.array(pose3d).transpose(1, 0, 2)  # (frame, marker, xyz)
    return pose3d, labels


def load_rig_mapping(rig_file: str, rig_name: str, marker_labels: List[str]) -> Tuple[List[str], List[List[int]]]:
    with open(rig_file, "r", encoding='utf-8') as f:
        rig_data = json.load(f)
    
    joint_names = list(rig_data[rig_name].keys())
    marker_idxs = [[marker_labels.index(label) for label in rig_data[rig_name][joint]] for joint in joint_names]

    return joint_names, marker_idxs


def apply_rig_format(pose3d: np.ndarray, joint_names: List[str], marker_idxs: List[List[int]]) -> np.ndarray:
    formatted_pose3d = np.zeros((pose3d.shape[0], len(joint_names), 3))
    for i, idxs in enumerate(marker_idxs):
        formatted_pose3d[:, i, :] = np.mean(pose3d[:, idxs, :], axis=1)
    return formatted_pose3d


def process_file(json_file: Path, rig_file: str, rig_name: str, output_root: Path):
    print(f"Converting {json_file} ...")
    marker_data = get_marker_data(json_file)
    time_range = get_time_range(marker_data)
    pose3d, marker_labels = get_pose_array(marker_data, time_range)
    joint_names, marker_idxs = load_rig_mapping(rig_file, rig_name, marker_labels)
    formatted_pose3d = apply_rig_format(pose3d, joint_names, marker_idxs)
    
    # Prepare output directory
    json_parts = list(json_file.parent.parts)
    lower_parts = [part.lower() for part in json_parts]
    if "json" in lower_parts:
        rel_parts = json_parts[lower_parts.index("json") + 1:]
    else:
        rel_parts = list(json_file.parent.parts)
    output_dir = output_root / Path(*rel_parts)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the formatted pose3d array
    output_filename = json_file.with_suffix('.npy').name
    np.save(output_dir / output_filename, formatted_pose3d)


def main():
    # parser
    parser = ArgumentParser()
    parser.add_argument("--rig", type=str, default='Human3.6M', help="Rig mapping to use")
    parser.add_argument("--input-root", type=Path, default=Path("./data/json"), help="JSON 数据根目录")
    parser.add_argument("--output-root", type=Path, default=None, help="NPY 输出根目录")
    args = parser.parse_args()
    output_root = args.output_root or (Path("./data/npy_original84") if args.rig == "ORIGINAL" else Path("./data/npy"))
    
    skaters = ['Skater_A', 'Skater_B', 'Skater_C', 'Skater_D']
    jumps = ['Axel', 'Comb', 'Flip', 'Lutz', 'Salchow', 'Loop', 'Toeloop']

    files = [args.input_root / skater / jump / f
             for skater in skaters
             for jump in jumps
             if (args.input_root / skater / jump).exists()
             for f in os.listdir(args.input_root / skater / jump) if f.endswith('.json')]

    rig_file = './utils/rig.json'
    for json_file in files:
        process_file(json_file, rig_file, args.rig, output_root)

    print(f"Successfully converted all JSON files to {output_root}.")

if __name__ == '__main__':
    main()
