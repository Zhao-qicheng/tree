from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def _as_tuple(names: Sequence[str]) -> Tuple[str, ...]:
    return tuple(str(name) for name in names)


@dataclass(slots=True)
class FlatOctree:
    """
    扁平化八叉树存储结构（Structure of Arrays）。

    通过多个并行的 numpy 数组保存节点信息，显著降低内存占用并支持 mmap 加载。
    """

    keypoint_names: Tuple[str, ...]
    node_parent_idx: np.ndarray  # shape: (N,)
    node_depth: np.ndarray  # shape: (N,)
    bboxes: np.ndarray  # shape: (N, K, 6)
    children_row_ptr: np.ndarray  # shape: (N + 1,)
    children_octants: np.ndarray  # shape: (M, K)
    children_indices: np.ndarray  # shape: (M,)
    frame_ids_row_ptr: np.ndarray  # shape: (N + 1,)
    frame_ids_data: np.ndarray  # shape: (F,)
    version: int = 1
    _npz_handle: Optional[np.lib.npyio.NpzFile] = None

    # ------------------------------------------------------------------
    # 构造 & 工厂方法
    # ------------------------------------------------------------------
    @classmethod
    def allocate(
        cls,
        keypoint_names: Sequence[str],
        num_nodes: int,
        *,
        num_children_entries: int,
        num_frame_entries: int,
        bbox_dtype: np.dtype = np.float32,
        depth_dtype: np.dtype = np.int16,
        frame_id_dtype: str = "U32",
    ) -> "FlatOctree":
        """
        根据所需容量创建一个空的扁平化八叉树实例。

        参数:
            keypoint_names: 关键点名称序列
            num_nodes: 节点数量
            num_children_entries: children CSR 总长度
            num_frame_entries: 帧ID总数
            bbox_dtype: 包围盒存储精度
            depth_dtype: depth字段精度，默认 int16 支持较深的树
            frame_id_dtype: 帧ID字符串 dtype
        """
        keypoint_tuple = _as_tuple(keypoint_names)
        num_keypoints = len(keypoint_tuple)

        node_parent_idx = np.full(num_nodes, -1, dtype=np.int32)
        node_depth = np.zeros(num_nodes, dtype=depth_dtype)
        bboxes = np.zeros((num_nodes, num_keypoints, 6), dtype=bbox_dtype)

        children_row_ptr = np.zeros(num_nodes + 1, dtype=np.int32)
        children_octants = np.zeros((num_children_entries, num_keypoints), dtype=np.uint8)
        children_indices = np.full(num_children_entries, -1, dtype=np.int32)

        frame_ids_row_ptr = np.zeros(num_nodes + 1, dtype=np.int32)
        frame_ids_data = np.empty(num_frame_entries, dtype=frame_id_dtype)

        return cls(
            keypoint_names=keypoint_tuple,
            node_parent_idx=node_parent_idx,
            node_depth=node_depth,
            bboxes=bboxes,
            children_row_ptr=children_row_ptr,
            children_octants=children_octants,
            children_indices=children_indices,
            frame_ids_row_ptr=frame_ids_row_ptr,
            frame_ids_data=frame_ids_data,
        )

    # ------------------------------------------------------------------
    # 基础属性
    # ------------------------------------------------------------------
    @property
    def num_nodes(self) -> int:
        return int(self.node_parent_idx.shape[0])

    @property
    def num_children_entries(self) -> int:
        return int(self.children_indices.shape[0])

    @property
    def num_frame_entries(self) -> int:
        return int(self.frame_ids_data.shape[0])

    # ------------------------------------------------------------------
    # 数据访问辅助
    # ------------------------------------------------------------------
    def get_children_indices(self, node_index: int) -> np.ndarray:
        start = int(self.children_row_ptr[node_index])
        end = int(self.children_row_ptr[node_index + 1])
        return self.children_indices[start:end]

    def get_children_octants(self, node_index: int) -> np.ndarray:
        start = int(self.children_row_ptr[node_index])
        end = int(self.children_row_ptr[node_index + 1])
        return self.children_octants[start:end]

    def iter_children(self, node_index: int) -> Iterable[tuple[np.ndarray, int]]:
        """
        迭代 node_index 的所有子节点，返回 (octant_vector, child_index)。
        """
        start = int(self.children_row_ptr[node_index])
        end = int(self.children_row_ptr[node_index + 1])
        for idx in range(start, end):
            yield self.children_octants[idx], int(self.children_indices[idx])

    def get_frame_ids(self, node_index: int) -> np.ndarray:
        start = int(self.frame_ids_row_ptr[node_index])
        end = int(self.frame_ids_row_ptr[node_index + 1])
        return self.frame_ids_data[start:end]

    def get_bbox(self, node_index: int) -> np.ndarray:
        return self.bboxes[node_index]

    # ------------------------------------------------------------------
    # 序列化 / 反序列化
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """
        保存扁平化八叉树为 .npz 文件。
        """
        np.savez(
            path,
            version=np.int32(self.version),
            keypoint_names=np.array(self.keypoint_names, dtype="U32"),
            node_parent_idx=self.node_parent_idx,
            node_depth=self.node_depth,
            bboxes=self.bboxes,
            children_row_ptr=self.children_row_ptr,
            children_octants=self.children_octants,
            children_indices=self.children_indices,
            frame_ids_row_ptr=self.frame_ids_row_ptr,
            frame_ids_data=self.frame_ids_data,
        )

    @classmethod
    def load(cls, path: str | Path, mmap_mode: Optional[str] = None) -> "FlatOctree":
        """
        从 .npz 文件加载扁平化八叉树。
        """
        npz = np.load(path, allow_pickle=False, mmap_mode=mmap_mode)
        keypoint_names = tuple(str(name) for name in npz["keypoint_names"])

        instance = cls(
            keypoint_names=keypoint_names,
            node_parent_idx=npz["node_parent_idx"],
            node_depth=npz["node_depth"],
            bboxes=npz["bboxes"],
            children_row_ptr=npz["children_row_ptr"],
            children_octants=npz["children_octants"],
            children_indices=npz["children_indices"],
            frame_ids_row_ptr=npz["frame_ids_row_ptr"],
            frame_ids_data=npz["frame_ids_data"],
            version=int(npz["version"]),
            _npz_handle=npz if mmap_mode else None,
        )
        if mmap_mode is None:
            # 未使用mmap时可立即关闭文件句柄
            npz.close()
            instance._npz_handle = None
        return instance

    def close(self) -> None:
        """
        关闭内部的 npz 句柄（若使用 mmap）。
        """
        if self._npz_handle is not None:
            self._npz_handle.close()
            self._npz_handle = None

    # ------------------------------------------------------------------
    # 上下文管理
    # ------------------------------------------------------------------
    def __enter__(self) -> "FlatOctree":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

