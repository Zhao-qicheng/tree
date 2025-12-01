"""
旋转工具模块：提供坐标系旋转功能以支持多树索引。
"""

from __future__ import annotations

from typing import Dict, List
import numpy as np


def get_rotation_matrix(axis: str, angle_deg: float) -> np.ndarray:
    """
    获取绕指定轴旋转的旋转矩阵（右手定则）。
    
    参数:
        axis: 旋转轴 ('x', 'y', 'z')
        angle_deg: 旋转角度（度）
    
    返回:
        3x3 旋转矩阵
    
    示例:
        >>> R = get_rotation_matrix('z', 90)
        >>> point = np.array([1, 0, 0])
        >>> rotated = R @ point
        >>> # [0, 1, 0]（X轴旋转到Y轴）
    """
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    
    if axis.lower() == 'x':
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ], dtype=np.float64)
    elif axis.lower() == 'y':
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ], dtype=np.float64)
    elif axis.lower() == 'z':
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ], dtype=np.float64)
    else:
        raise ValueError(f"不支持的旋转轴: {axis}，必须是 'x', 'y', 或 'z'")


def rotate_keypoints(keypoints: Dict[str, np.ndarray], 
                     rotation_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """
    对所有关键点应用旋转变换。
    
    参数:
        keypoints: 关键点字典 {关节名: 3D坐标}
        rotation_matrix: 3x3 旋转矩阵
    
    返回:
        旋转后的关键点字典
    
    注意:
        - hip原点保持为[0,0,0]（旋转后仍为原点）
        - 所有相对坐标都会被旋转
    """
    rotated = {}
    for name, pos in keypoints.items():
        # 应用旋转：R @ v
        rotated[name] = rotation_matrix @ pos
    return rotated


def rotate_inverse_keypoints(keypoints: Dict[str, np.ndarray],
                             rotation_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """
    应用逆旋转（用于将旋转后的坐标恢复到原始坐标系）。
    
    参数:
        keypoints: 旋转后的关键点字典
        rotation_matrix: 原始旋转矩阵
    
    返回:
        原始坐标系下的关键点
    
    注意:
        旋转矩阵是正交矩阵，其逆矩阵等于转置：R^(-1) = R^T
    """
    inverse_matrix = rotation_matrix.T
    return rotate_keypoints(keypoints, inverse_matrix)


class RotationConfig:
    """旋转配置类，封装单个树的旋转参数。"""
    
    def __init__(self, tree_id: int, axis: str, angle: float):
        """
        参数:
            tree_id: 树的唯一标识符（0为原始坐标系）
            axis: 旋转轴
            angle: 旋转角度（度）
        """
        self.tree_id = tree_id
        self.axis = axis
        self.angle = angle
        self.rotation_matrix = get_rotation_matrix(axis, angle)
    
    def __repr__(self) -> str:
        return f"RotationConfig(tree_id={self.tree_id}, axis='{self.axis}', angle={self.angle}°)"
    
    def rotate(self, keypoints: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """对关键点应用此配置的旋转。"""
        return rotate_keypoints(keypoints, self.rotation_matrix)
    
    def rotate_inverse(self, keypoints: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """应用逆旋转。"""
        return rotate_inverse_keypoints(keypoints, self.rotation_matrix)
    
    def get_model_filename(self, base_name: str = "model") -> str:
        """
        生成此树的模型文件名。
        
        示例:
            tree_id=0: "model_tree0_z0.tree"
            tree_id=1: "model_tree1_z30.tree"
        """
        return f"{base_name}_tree{self.tree_id}_{self.axis}{int(self.angle)}.tree"
    
    def get_metadata_filename(self, base_name: str = "model") -> str:
        """生成此树的元数据文件名。"""
        return f"{base_name}_tree{self.tree_id}_{self.axis}{int(self.angle)}.pkl"


def create_default_rotation_configs() -> List[RotationConfig]:
    """
    创建默认的5棵树旋转配置。
    
    返回:
        RotationConfig列表
    """
    configs = [
        RotationConfig(tree_id=0, axis='z', angle=0),    # 原始坐标系
        RotationConfig(tree_id=1, axis='z', angle=30),   # Z轴旋转30°
        RotationConfig(tree_id=2, axis='z', angle=60),   # Z轴旋转60°
        RotationConfig(tree_id=3, axis='y', angle=30),   # Y轴旋转30°
        RotationConfig(tree_id=4, axis='x', angle=30),   # X轴旋转30°
    ]
    return configs


def create_custom_rotation_configs(config_dicts: List[Dict]) -> List[RotationConfig]:
    """
    从配置字典列表创建旋转配置。
    
    参数:
        config_dicts: 配置字典列表，每个字典包含 'axis' 和 'angle' 键
    
    示例:
        >>> configs = create_custom_rotation_configs([
        ...     {"axis": "z", "angle": 0},
        ...     {"axis": "z", "angle": 45},
        ... ])
    """
    return [
        RotationConfig(tree_id=i, axis=cfg['axis'], angle=cfg['angle'])
        for i, cfg in enumerate(config_dicts)
    ]


# 用于测试的辅助函数
def test_rotation():
    """测试旋转功能是否正常工作。"""
    print("=" * 60)
    print("测试旋转功能")
    print("=" * 60)
    
    # 测试1: 基本旋转
    print("\n测试1: Z轴旋转90°")
    R = get_rotation_matrix('z', 90)
    point = np.array([1.0, 0.0, 0.0])
    rotated = R @ point
    print(f"原始点: {point}")
    print(f"旋转后: {rotated}")
    print(f"预期: [0, 1, 0]")
    assert np.allclose(rotated, [0, 1, 0]), "Z轴旋转测试失败"
    
    # 测试2: 关键点旋转
    print("\n测试2: 关键点旋转")
    keypoints = {
        'hip': np.array([0.0, 0.0, 0.0]),
        'chest': np.array([1.0, 2.0, 3.0]),
    }
    config = RotationConfig(tree_id=1, axis='z', angle=90)
    rotated_kp = config.rotate(keypoints)
    print(f"原始chest: {keypoints['chest']}")
    print(f"旋转后chest: {rotated_kp['chest']}")
    
    # 测试3: 逆旋转
    print("\n测试3: 逆旋转恢复原始坐标")
    recovered = config.rotate_inverse(rotated_kp)
    print(f"恢复后chest: {recovered['chest']}")
    assert np.allclose(recovered['chest'], keypoints['chest']), "逆旋转测试失败"
    
    # 测试4: 默认配置
    print("\n测试4: 默认旋转配置")
    configs = create_default_rotation_configs()
    for cfg in configs:
        print(f"  {cfg}")
        print(f"    模型文件: {cfg.get_model_filename()}")
    
    print("\n✅ 所有测试通过！")


if __name__ == "__main__":
    test_rotation()
