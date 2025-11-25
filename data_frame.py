from bvh import Bvh
import numpy as np
import math

# 这是标准的 BVH 坐标系：Y 向上，Z 向前，X 向左（右手系）

# 全局变量：缓存解析结果以提高效率，以文件路径为键
_bvh_cache = {}


def load_bvh_file(bvh_file='walk.bvh'):
    """
    加载并解析 BVH 文件，支持多文件缓存。
    
    参数:
        bvh_file: BVH文件路径（字符串）
    
    返回:
        包含解析结果的字典，包括：
        - mocap: Bvh对象
        - all_joints: 关节名称列表
        - root_name: 根节点名称
        - joint_offsets: 关节偏移量字典
        - parent_map: 父子关系映射
        - joint_order: 关节顺序列表
        - channel_map: 通道映射字典
    """
    global _bvh_cache
    
    # 标准化文件路径（解决相对路径问题）
    import os
    bvh_file = os.path.abspath(bvh_file)
    
    # 如果文件已缓存，直接返回
    if bvh_file in _bvh_cache:
        return _bvh_cache[bvh_file]
    
    # 检查文件是否存在
    if not os.path.exists(bvh_file):
        raise FileNotFoundError(f"BVH文件不存在: {bvh_file}")
    
    # 解析并缓存新文件
    cache = {}
    
    with open(bvh_file, 'r') as f:
        cache['mocap'] = Bvh(f.read())
    
    cache['all_joints'] = cache['mocap'].get_joints_names()
    
    with open(bvh_file, 'r') as f:
        cache['root_name'] = next(
            (line.split()[1] for line in f if line.strip().startswith('ROOT')),
            cache['all_joints'][0]
        )
    
    # 解析层次结构
    cache['joint_offsets'], cache['parent_map'], cache['joint_order'] = parse_bvh_hierarchy(bvh_file)
    
    # 构建通道映射
    cache['channel_map'] = {}
    channel_idx = 0
    cache['channel_map'][cache['root_name']] = {'start': 0, 'count': 6}
    channel_idx = 6
    
    for joint_name in cache['all_joints']:
        if joint_name != cache['root_name']:
            cache['channel_map'][joint_name] = {'start': channel_idx, 'count': 3}
            channel_idx += 3
    
    # 缓存结果
    _bvh_cache[bvh_file] = cache
    return cache

def parse_bvh_hierarchy(bvh_file):
    """解析 BVH 文件的层次结构"""
    offsets = {}
    parent_map = {}
    joint_order = []
    joint_stack = []
    
    with open(bvh_file, 'r') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('ROOT') or stripped.startswith('JOINT'):
                joint_name = stripped.split()[1]
                if joint_name not in joint_order:
                    joint_order.append(joint_name)
                if joint_stack:
                    parent_map[joint_name] = joint_stack[-1]
                joint_stack.append(joint_name)
            elif stripped.startswith('OFFSET'):
                parts = stripped.split()
                if len(parts) == 4 and joint_stack:
                    offsets[joint_stack[-1]] = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            elif stripped == '}':
                if joint_stack:
                    joint_stack.pop()
    return offsets, parent_map, joint_order

def euler_zyx_to_matrix(angles_deg):
    """ZYX 欧拉角转旋转矩阵"""
    z, y, x = np.deg2rad(angles_deg)
    cz, sz = math.cos(z), math.sin(z)
    cy, sy = math.cos(y), math.sin(y)
    cx, sx = math.cos(x), math.sin(x)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return Rz @ Ry @ Rx

def calculate_joint_positions(frame_index, bvh_file='walk.bvh'):
    """
    计算指定BVH文件中指定帧的所有关节位置
    
    参数:
        frame_index: 帧索引（从0开始）
        bvh_file: BVH文件路径
    
    返回:
        关节名称 -> 3D位置坐标的字典
    """
    cache = load_bvh_file(bvh_file)
    mocap = cache['mocap']
    all_joints = cache['all_joints']
    root_name = cache['root_name']
    joint_offsets = cache['joint_offsets']
    parent_map = cache['parent_map']
    channel_map = cache['channel_map']
    
    # 检查帧索引有效性
    if frame_index < 0 or frame_index >= mocap.nframes:
        raise ValueError(f"帧索引 {frame_index} 超出范围 [0, {mocap.nframes-1}]")
    
    # 获取帧数据
    frame_values = [float(x) for x in mocap.frames[frame_index]]
    
    # 计算所有关节位置
    joint_positions = {}
    joint_rotations = {}
    
    # 根节点
    ch = channel_map[root_name]
    root_pos = np.array(frame_values[ch['start']:ch['start']+3])
    root_rot_angles = np.array(frame_values[ch['start']+3:ch['start']+6])
    root_rot = euler_zyx_to_matrix(root_rot_angles)
    joint_positions[root_name] = root_pos
    joint_rotations[root_name] = root_rot
    
    # 按照 all_joints 的顺序计算其他关节
    for joint_name in all_joints:
        if joint_name == root_name:
            continue
        
        if joint_name not in channel_map:
            continue
        
        # 找到父关节
        if joint_name in parent_map:
            parent_name = parent_map[joint_name]
            if parent_name in joint_positions:
                parent_pos = joint_positions[parent_name]
                parent_rot = joint_rotations[parent_name]
            else:
                continue
        else:
            parent_pos = root_pos
            parent_rot = root_rot
        
        # 计算当前关节
        ch = channel_map[joint_name]
        ch_start = ch['start']
        rot_angles = np.array(frame_values[ch_start:ch_start+3])
        rot = euler_zyx_to_matrix(rot_angles)
        offset = joint_offsets.get(joint_name, np.array([0, 0, 0]))
        pos = parent_pos + parent_rot @ offset
        rot = parent_rot @ rot
        
        joint_positions[joint_name] = pos
        joint_rotations[joint_name] = rot
    
    return joint_positions

def get_joint_position(frame_index, joint_name, bvh_file='walk.bvh', relative_to_hips=True, unit='original'):
    """
    获取指定BVH文件中指定帧和指定关节的位置数据
    
    参数:
        frame_index: 帧索引（从 0 开始）
        joint_name: 关节名称（字符串，使用BVH原始名称如 'Hips', 'LeftHand' 等）
        bvh_file: BVH文件路径（默认 'walk.bvh'）
        relative_to_hips: 是否返回相对于 Hips 的局部坐标（默认 True）
        unit: 单位，'original'（原始单位）、'cm'（厘米）、'm'（米），默认 'original'
    
    返回:
        numpy.ndarray: 关节的 3D 位置坐标 [X, Y, Z]
    
    示例:
        # 获取 walk.bvh 第 0 帧 Hips 的位置（相对于 Hips，应该是 [0, 0, 0]）
        pos = get_joint_position(0, 'Hips', bvh_file='data/walk.bvh')
        
        # 获取第 10 帧 LeftHand 的世界坐标位置
        pos = get_joint_position(10, 'LeftHand', bvh_file='data/walk.bvh', relative_to_hips=False)
        
        # 获取另一个BVH文件的数据
        pos = get_joint_position(5, 'Neck', bvh_file='data/01_01.bvh', unit='m')
    """
    cache = load_bvh_file(bvh_file)
    
    # 检查关节名称是否存在
    if joint_name not in cache['all_joints']:
        available = ', '.join(cache['all_joints'][:10])
        raise ValueError(f"关节 '{joint_name}' 在文件 '{bvh_file}' 中不存在。可用关节: {available}...")
    
    # 计算所有关节位置
    joint_positions = calculate_joint_positions(frame_index, bvh_file)
    
    # 获取世界坐标
    if joint_name not in joint_positions:
        raise ValueError(f"无法计算关节 '{joint_name}' 的位置")
    
    world_pos = joint_positions[joint_name]
    
    # 转换为相对坐标
    if relative_to_hips:
        hips_pos = joint_positions[cache['root_name']]
        pos = world_pos - hips_pos
    else:
        pos = world_pos
    
    # 单位转换
    if unit == 'm':
        # 假设原始单位是厘米，转换为米
        pos = pos / 100.0
    elif unit == 'cm':
        # 保持厘米（如果原始单位不是厘米，可能需要调整）
        pass
    elif unit == 'original':
        # 保持原始单位
        pass
    else:
        raise ValueError(f"不支持的单位: {unit}。支持: 'original', 'cm', 'm'")
    
    return pos

def get_all_joints(frame_index, bvh_file='walk.bvh', relative_to_hips=True, unit='original'):
    """
    获取指定BVH文件中指定帧的所有关节位置
    
    参数:
        frame_index: 帧索引
        bvh_file: BVH文件路径（默认 'walk.bvh'）
        relative_to_hips: 是否返回相对于 Hips 的局部坐标
        unit: 单位，'original'（原始单位）、'cm'（厘米）、'm'（米）
    
    返回:
        dict: 关节名称 -> 位置坐标的字典
    """
    cache = load_bvh_file(bvh_file)
    joint_positions = calculate_joint_positions(frame_index, bvh_file)
    
    result = {}
    hips_pos = joint_positions[cache['root_name']] if relative_to_hips else np.array([0, 0, 0])
    
    for joint_name, world_pos in joint_positions.items():
        pos = world_pos - hips_pos if relative_to_hips else world_pos
        
        # 单位转换
        if unit == 'm':
            pos = pos / 100.0
        elif unit == 'cm':
            pass
        elif unit == 'original':
            pass
        
        result[joint_name] = pos
    
    return result