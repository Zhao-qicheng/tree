"""
动作八叉树节点定义与辅助方法。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Iterator, MutableMapping, Optional, Tuple

import config
from data_structures import BoundingBox, MultiPointOctant


class ActionTreeNode:
    """
    八叉树节点，用于帧检索系统。
    维护关键点包围盒、帧ID列表及子节点。
    """

    __slots__ = (
        "depth",
        "parent",
        "children",
        "bboxes",
        "frame_ids",  # 存储帧ID列表而不是标签统计
        "keypoint_names",
    )

    def __init__(
        self,
        depth: int,
        bboxes: Dict[str, BoundingBox],
        parent: Optional["ActionTreeNode"] = None,
        keypoint_names: Optional[Tuple[str, ...]] = None,
    ) -> None:
        self.depth = depth
        self.parent = parent
        self.children: Dict[MultiPointOctant, ActionTreeNode] = {}
        self.bboxes = bboxes
        self.frame_ids: list[str] = []  # 存储落在该节点的所有帧ID
        if keypoint_names is None:
            keypoint_names = tuple(config.OCTREE_KEYPOINT_NAMES)
        self.keypoint_names: Tuple[str, ...] = keypoint_names

    # ======================== 子节点管理 ========================

    def get_child(self, octants: MultiPointOctant) -> Optional["ActionTreeNode"]:
        return self.children.get(octants)

    def get_or_create_child(
        self,
        octants: MultiPointOctant,
    ) -> "ActionTreeNode":
        """
        获取或创建指定八分体组合的子节点。

        Args:
            octants: 多关键点 octant 编码组合，长度与用于八叉树的关键点数量一致（不包括Hips）
        """
        child = self.children.get(octants)
        if child is not None:
            return child

        if len(octants) != len(self.keypoint_names):
            raise ValueError(
                f"octants 长度应为 {len(self.keypoint_names)}，当前为 {len(octants)}"
            )

        child_bboxes: Dict[str, BoundingBox] = {}
        for idx, name in enumerate(self.keypoint_names):
            bbox = self.bboxes[name]
            child_bboxes[name] = bbox.subdivide(octants[idx])

        child = ActionTreeNode(
            depth=self.depth + 1,
            bboxes=child_bboxes,
            parent=self,
            keypoint_names=self.keypoint_names,
        )
        self.children[octants] = child
        return child

    def iter_children(self) -> Iterator[Tuple[MultiPointOctant, "ActionTreeNode"]]:
        return iter(self.children.items())

    def is_leaf(self) -> bool:
        return not self.children

    # ======================== 帧ID管理 ========================

    def add_frame(self, frame_id: str) -> None:
        """为当前节点添加一个帧ID。"""
        if frame_id not in self.frame_ids:
            self.frame_ids.append(frame_id)
    
    def get_frame_ids(self) -> list[str]:
        """获取当前节点存储的所有帧ID。"""
        return self.frame_ids

    # ======================== 序列化辅助 ========================

    def to_serializable(self) -> Dict:
        """递归生成可序列化字典。"""
        return {
            "depth": self.depth,
            "frame_ids": self.frame_ids,
            "keypoint_names": list(self.keypoint_names),
            "bboxes": {
                name: bbox.to_tuple()
                for name, bbox in self.bboxes.items()
            },
            "children": {
                ",".join(map(str, key)): child.to_serializable()
                for key, child in self.children.items()
            },
        }

    @staticmethod
    def from_serializable(
        data: Dict,
        parent: Optional["ActionTreeNode"] = None,
    ) -> "ActionTreeNode":
        """从序列化字典重建节点。"""
        bboxes = {
            name: BoundingBox.from_tuple(values)
            for name, values in data["bboxes"].items()
        }
        node = ActionTreeNode(
            depth=int(data["depth"]),
            bboxes=bboxes,
            parent=parent,
            keypoint_names=tuple(data.get("keypoint_names", config.OCTREE_KEYPOINT_NAMES)),
        )
        node.frame_ids = list(data.get("frame_ids", []))
        for key_str, child_data in data.get("children", {}).items():
            if isinstance(key_str, (list, tuple)):
                key_tuple = tuple(int(v) for v in key_str)
            else:
                parts = [part.strip() for part in str(key_str).split(",") if part.strip()]
                key_tuple = tuple(int(part) for part in parts)
            child = ActionTreeNode.from_serializable(child_data, parent=node)
            node.children[key_tuple] = child
        return node

