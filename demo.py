"""
最小化演示脚本：展示如何构建八叉树、插入样本并执行推理。
"""

from __future__ import annotations

from pathlib import Path

from octree_builder import create_root_node, finalize_tree, insert_sample, save_tree
from inference import predict_action


def build_demo_tree():
    root = create_root_node()

    # 简单构造两类动作的示例样本（仅示意）
    standing = {
        "hips": [0.0, 0.0, 0.0],
        "left_wrist": [0.2, 0.6, 0.0],
        "right_wrist": [-0.2, 0.6, 0.0],
        "neck": [0.0, 0.8, 0.0],
        "left_ankle": [0.1, -0.8, 0.0],
        "right_ankle": [-0.1, -0.8, 0.0],
    }
    raising_hand = {
        "hips": [0.0, 0.0, 0.0],
        "left_wrist": [0.3, 1.2, 0.0],
        "right_wrist": [-0.2, 0.5, 0.0],
        "neck": [0.0, 0.8, 0.0],
        "left_ankle": [0.1, -0.8, 0.0],
        "right_ankle": [-0.1, -0.8, 0.0],
    }

    insert_sample(root, standing, "standing")
    insert_sample(root, raising_hand, "raising_hand")
    finalize_tree(root)
    return root


def main():
    root = build_demo_tree()

    query = {
        "hips": [0.0, 0.0, 0.0],
        "left_wrist": [0.25, 1.1, 0.0],
        "right_wrist": [-0.25, 0.5, 0.0],
        "neck": [0.0, 0.8, 0.0],
        "left_ankle": [0.1, -0.8, 0.0],
        "right_ankle": [-0.1, -0.8, 0.0],
    }
    result = predict_action(root, query)
    print("预测动作:", result.label)
    print("遍历深度:", result.terminated_at_depth, "是否回退:", result.fallback_triggered)

    output_path = Path("demo_tree.json")
    save_tree(root, output_path.as_posix())
    print(f"模型已保存至 {output_path}")


if __name__ == "__main__":
    main()

