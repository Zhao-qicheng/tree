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
    八叉树节点，维护关键点包围盒、样本统计及子节点。
    """

    __slots__ = (
        "depth",
        "parent",
        "children",
        "bboxes",
        "sample_counts",
        "total_samples",
        "resolved_label",
    )

    def __init__(
        self,
        depth: int,
        bboxes: Dict[str, BoundingBox],
        parent: Optional["ActionTreeNode"] = None,
    ) -> None:
        self.depth = depth
        self.parent = parent
        self.children: Dict[MultiPointOctant, ActionTreeNode] = {}
        self.bboxes = bboxes
        self.sample_counts: Dict[str, int] = defaultdict(int)
        self.total_samples: int = 0
        self.resolved_label: Optional[str] = None

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
            octants: 多关键点 octant 编码组合，长度与关键点数量一致
        """
        child = self.children.get(octants)
        if child is not None:
            return child

        if len(octants) != len(config.KEYPOINT_NAMES):
            raise ValueError(
                f"octants 长度应为 {len(config.KEYPOINT_NAMES)}，当前为 {len(octants)}"
            )

        child_bboxes: Dict[str, BoundingBox] = {}
        for idx, name in enumerate(config.KEYPOINT_NAMES):
            bbox = self.bboxes[name]
            child_bboxes[name] = bbox.subdivide(octants[idx])

        child = ActionTreeNode(
            depth=self.depth + 1,
            bboxes=child_bboxes,
            parent=self,
        )
        self.children[octants] = child
        return child

    def iter_children(self) -> Iterator[Tuple[MultiPointOctant, "ActionTreeNode"]]:
        return iter(self.children.items())

    def is_leaf(self) -> bool:
        return not self.children

    # ======================== 样本统计 ========================

    def record_sample(self, label: str) -> None:
        """为当前节点记录一个样本标签。"""
        self.total_samples += 1
        self.sample_counts[label] += 1

    def resolve_label(self) -> Optional[str]:
        """
        根据样本统计确定节点标签。

        返回多数票标签；若无样本则保持原标签。
        """
        if not self.sample_counts:
            return self.resolved_label

        # 多数票，若出现并列则按标签字典序稳定选择。
        resolved = max(
            self.sample_counts.items(),
            key=lambda item: (item[1], item[0]),
        )[0]
        self.resolved_label = resolved
        return resolved

    # ======================== 序列化辅助 ========================

    def to_serializable(self) -> Dict:
        """递归生成可 JSON 化字典。"""
        return {
            "depth": self.depth,
            "resolved_label": self.resolved_label,
            "total_samples": self.total_samples,
            "sample_counts": dict(self.sample_counts),
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
        """从 JSON 化字典重建节点。"""
        bboxes = {
            name: BoundingBox.from_tuple(values)
            for name, values in data["bboxes"].items()
        }
        node = ActionTreeNode(
            depth=int(data["depth"]),
            bboxes=bboxes,
            parent=parent,
        )
        node.resolved_label = data.get("resolved_label")
        node.total_samples = int(data.get("total_samples", 0))
        node.sample_counts.update(
            {label: int(count) for label, count in data.get("sample_counts", {}).items()}
        )
        for key_str, child_data in data.get("children", {}).items():
            if isinstance(key_str, (list, tuple)):
                key_tuple = tuple(int(v) for v in key_str)
            else:
                parts = [part.strip() for part in str(key_str).split(",") if part.strip()]
                key_tuple = tuple(int(part) for part in parts)
            child = ActionTreeNode.from_serializable(child_data, parent=node)
            node.children[key_tuple] = child
        return node

