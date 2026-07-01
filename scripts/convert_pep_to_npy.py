"""
将 PoseStudio (.pep) 文件转换为八叉树系统可处理的格式。

使用方法:
    python scripts/convert_pep_to_npy.py --input data/下楼梯.pep --output output/下楼梯
"""

import xml.etree.ElementTree as ET
import numpy as np
import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data_structures import BodyKeypoints, BoundingBox, normalize_to_hip
from flat_octree import FlatOctree


def parse_rotquat(rotquat_str: str) -> np.ndarray:
    """解析四元数字符串 (w x y z)"""
    parts = [float(x) for x in rotquat_str.strip().split()]
    return np.array(parts)  # [w, x, y, z]


def parse_trans(trans_str: str) -> np.ndarray:
    """解析平移向量字符串 (x y z)，返回毫米单位"""
    parts = [float(x) for x in trans_str.strip().split()]
    return np.array(parts)  # [x, y, z]


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """将四元数 (w, x, y, z) 转换为 3x3 旋转矩阵"""
    w, x, y, z = q
    
    # 归一化
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm > 0:
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # 转换为旋转矩阵
    xx, xy, xz = x*x, x*y, x*z
    yy, yz, zz = y*y, y*z, z*z
    wx, wy, wz = w*x, w*y, w*z
    
    R = np.array([
        [1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)],
        [2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)]
    ])
    
    return R


def build_joint_tree(node: ET.Element, parent_transform: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    递归构建关节树并计算世界坐标。
    
    返回: {关节名: 世界坐标 (3,)}
    """
    joints = {}
    
    # 获取当前节点的变换
    name = node.get('name', '')
    rotquat = parse_rotquat(node.get('rotquat', '1 0 0 0'))
    trans = parse_trans(node.get('trans', '0 0 0'))
    
    # 计算局部变换矩阵 (4x4)
    R = quat_to_matrix(rotquat)
    T_local = np.eye(4)
    T_local[:3, :3] = R
    T_local[:3, 3] = trans  # 毫米单位
    
    # 计算世界变换
    if parent_transform is None:
        T_world = T_local
    else:
        T_world = parent_transform @ T_local
    
    # 记录当前关节的世界坐标 (hip 在根节点)
    world_pos = T_world[:3, 3]
    
    # 只记录关键的骨骼关节（跳过 _end_ 标记的末端节点）
    if not name.endswith('_end_bb_') and '_end_' not in name:
        # 存储关节名到世界坐标的映射
        joints[name] = world_pos
    
    # 递归处理子节点
    for child in node:
        child_joints = build_joint_tree(child, T_world)
        joints.update(child_joints)
    
    return joints


# PoseStudio 到 Human3.6M 的关节名映射
JOINT_NAME_MAPPING = {
    # 髋部（根节点）
    'hips_bb_': 'hip',
    
    # 左腿
    'leftupleg_bb_': 'lHip',
    'leftleg_bb_': 'lKnee',
    'leftfoot_bb_': 'lAnkle',
    
    # 右腿
    'rightupleg_bb_': 'rHip',
    'rightleg_bb_': 'rKnee',
    'rightfoot_bb_': 'rAnkle',
    
    # 脊柱/躯干
    'spine_bb_': 'spine',
    'spine1_bb_': 'spine',
    'spine2_bb_': 'chest',
    
    # 颈部和头部
    'neck_bb_': 'neck',
    'head_bb_': 'head',
    
    # 左臂
    'leftshoulder_bb_': 'lShoulder',
    'leftarm_bb_': 'lElbow',
    'leftforearm_bb_': 'lWrist',
    
    # 右臂
    'rightshoulder_bb_': 'rShoulder',
    'rightarm_bb_': 'rElbow',
    'rightforearm_bb_': 'rWrist',
}


def convert_pep_to_keypoints(pep_file: str) -> Dict[str, np.ndarray]:
    """
    将 .pep 文件转换为 Human3.6M 格式的关节坐标。
    
    返回: {关节名: 坐标 (3,)}，坐标以 hip 为原点，单位毫米
    """
    # 解析 XML
    tree = ET.parse(pep_file)
    root = tree.getroot()
    
    # 找到 pose 节点下的第一个 node（通常是 __GroupRoot）
    pose_node = root.find('.//pose')
    if pose_node is None:
        raise ValueError(f"找不到 <pose> 节点在文件 {pep_file}")
    
    # 获取根节点（通常是 __GroupRoot 或 __SkeletonRoot001）
    root_joint = pose_node.find('node')
    if root_joint is None:
        raise ValueError(f"找不到根关节节点")
    
    # 递归构建关节树
    raw_joints = build_joint_tree(root_joint)
    
    # 映射到 Human3.6M 格式
    keypoints = {}
    for pep_name, h36m_name in JOINT_NAME_MAPPING.items():
        if pep_name in raw_joints:
            keypoints[h36m_name] = raw_joints[pep_name]
    
    # 检查是否所有必需关节都存在
    missing = set(config.KEYPOINT_NAMES) - set(keypoints.keys())
    if missing:
        print(f"警告: 缺少以下关节: {missing}")
        # 尝试从可用关节推断
        keypoints = fill_missing_joints(keypoints, raw_joints)
    
    # 归一化：以 hip 为原点
    hip_pos = keypoints.get('hip', np.zeros(3))
    for name in keypoints:
        keypoints[name] = keypoints[name] - hip_pos
    
    return keypoints


def fill_missing_joints(keypoints: Dict[str, np.ndarray], 
                       raw_joints: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """通过插值填充缺失的关节"""
    result = dict(keypoints)
    
    # 如果没有 hip，使用 hips_bb_ 或从子节点推断
    if 'hip' not in result:
        if 'hips_bb_' in raw_joints:
            result['hip'] = raw_joints['hips_bb_']
        else:
            # 从左右髋部平均推断
            lhip = raw_joints.get('leftupleg_bb_', np.zeros(3))
            rhip = raw_joints.get('rightupleg_bb_', np.zeros(3))
            result['hip'] = (lhip + rhip) / 2
    
    # 如果没有 spine，从 hips 和 chest 插值
    if 'spine' not in result:
        hip = result.get('hip', np.zeros(3))
        chest = result.get('chest', raw_joints.get('spine2_bb_', hip))
        result['spine'] = hip + (chest - hip) * 0.5
    
    # 如果没有 chest，从 spine 和 neck 插值
    if 'chest' not in result:
        spine = result.get('spine', result['hip'])
        neck = result.get('neck', raw_joints.get('neck_bb_', spine))
        result['chest'] = spine + (neck - spine) * 0.6
    
    # 填充其他缺失的关节
    for name in config.KEYPOINT_NAMES:
        if name not in result:
            # 默认设为 hip 位置（避免 NaN）
            result[name] = result.get('hip', np.zeros(3))
            print(f"  使用默认值填充: {name}")
    
    return result


def save_outputs(keypoints: Dict[str, np.ndarray], output_dir: str, pose_name: str):
    """保存所有输出格式"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 按 Human3.6M 顺序构建 (17, 3) 数组
    pose_array = np.zeros((17, 3), dtype=np.float64)
    for i, name in enumerate(config.KEYPOINT_NAMES):
        pose_array[i] = keypoints.get(name, np.zeros(3))
    
    # 保存 pose_3d.npy (17, 3)
    npy_path = os.path.join(output_dir, 'pose_3d.npy')
    np.save(npy_path, pose_array)
    print(f"  保存: {npy_path}")
    
    # 保存 pose_sequence.npy (1, 17, 3)
    seq_path = os.path.join(output_dir, 'pose_sequence.npy')
    seq_array = pose_array.reshape(1, 17, 3)
    np.save(seq_path, seq_array)
    print(f"  保存: {seq_path}")
    
    # 保存 keypoints.json
    json_path = os.path.join(output_dir, 'keypoints.json')
    json_data = {name: coord.round(6).tolist() for name, coord in keypoints.items()}
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"  保存: {json_path}")
    
    # 保存 pose_skeleton.npz (兼容 skeleton 格式)
    skeleton_path = os.path.join(output_dir, 'pose_skeleton.npz')
    np.savez_compressed(skeleton_path, reconstruction=seq_array)
    print(f"  保存: {skeleton_path}")
    
    # 保存 pose_octree.npz (简化的八叉树格式)
    octree = create_simple_octree(keypoints, pose_name)
    octree_path = os.path.join(output_dir, 'pose_octree.npz')
    octree.save(octree_path)
    print(f"  保存: {octree_path}")
    
    return {
        'npy': npy_path,
        'sequence': seq_path,
        'json': json_path,
        'skeleton': skeleton_path,
        'octree': octree_path
    }


def create_simple_octree(keypoints: Dict[str, np.ndarray], pose_name: str) -> FlatOctree:
    """创建一个简化的单节点八叉树，包含这个姿态"""
    tree = FlatOctree()
    
    # 设置关键点名称（排除 hip）
    tree.keypoint_names = list(config.OCTREE_KEYPOINT_NAMES)
    
    # 单个节点
    tree.num_nodes = 1
    tree.node_depth = np.array([0], dtype=np.int8)
    tree.node_parent_idx = np.array([-1], dtype=np.int32)
    tree.node_path_codes = np.array(['root'], dtype=object)
    
    # 为每个关键点创建包围盒 (简化：以关键点为中心的小盒子)
    num_kp = len(config.KEYPOINT_NAMES)
    bboxes = np.zeros((1, num_kp, 6), dtype=np.float64)
    
    for i, name in enumerate(config.KEYPOINT_NAMES):
        coord = keypoints.get(name, np.zeros(3))
        # 创建小包围盒（±1mm）
        bboxes[0, i] = [coord[0]-1, coord[1]-1, coord[2]-1, 
                        coord[0]+1, coord[1]+1, coord[2]+1]
    
    tree.bboxes = bboxes
    
    # 没有子节点
    tree.children_start = np.array([0, 0], dtype=np.int32)
    tree.children_keys = np.zeros((0, 2), dtype=np.uint8)
    tree.children_indices = np.array([], dtype=np.int32)
    
    # 帧 ID
    tree.frame_ids_start = np.array([0, 1], dtype=np.int32)
    tree.frame_ids_data = np.array([f"{pose_name}_frame_0"], dtype=object)
    
    return tree


def main():
    parser = argparse.ArgumentParser(description='转换 PoseStudio .pep 文件为八叉树格式')
    parser.add_argument('--input', '-i', type=str, required=True, help='输入 .pep 文件路径')
    parser.add_argument('--output', '-o', type=str, default=None, help='输出目录')
    args = parser.parse_args()
    
    # 确定输出目录
    if args.output is None:
        pose_name = Path(args.input).stem
        args.output = os.path.join('output', pose_name)
    
    print(f"转换文件: {args.input}")
    print(f"输出目录: {args.output}")
    
    # 转换
    keypoints = convert_pep_to_keypoints(args.input)
    
    # 打印关节位置摘要
    print("\n关节坐标摘要 (hip 为原点):")
    for name in config.KEYPOINT_NAMES:
        coord = keypoints.get(name, np.zeros(3))
        print(f"  {name:12s}: [{coord[0]:8.2f}, {coord[1]:8.2f}, {coord[2]:8.2f}]")
    
    # 保存所有格式
    print(f"\n保存输出文件到: {args.output}")
    paths = save_outputs(keypoints, args.output, Path(args.input).stem)
    
    print("\n转换完成！查看命令:")
    print(f"  静态查看: python utils/visualize_npy.py --path {paths['npy']} --frame 0")
    print(f"  动画查看: python utils/visualize_npy.py --path {paths['sequence']} --center")


if __name__ == '__main__':
    main()
