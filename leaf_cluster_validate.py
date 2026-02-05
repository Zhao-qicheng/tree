"""
八叉树叶节点聚类与跨树验证。

对树 A 的叶节点进行 K-Means 聚类，加载树 B 的元数据，
用树 B 的帧在树 A 上查询得到所属聚类，评估与真实动作标签的一致性。
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans

from flat_octree import FlatOctree
from octree_builder import load_tree, load_metadata
from query import find_leaf_node_for_query
from data_structures import FrameMetadata


def get_leaf_nodes(flat: FlatOctree) -> List[int]:
    """
    遍历所有节点，返回叶节点索引列表。
    叶节点判定：children_start[i] == children_start[i+1] 表示无子节点。
    """
    leaf_indices = []
    for i in range(flat.num_nodes):
        if flat.children_start[i] == flat.children_start[i + 1]:
            leaf_indices.append(i)
    return leaf_indices


def get_leaf_features(flat: FlatOctree, node_indices: List[int]) -> np.ndarray:
    """
    提取叶节点 bbox 特征：中心点与尺寸，展平为 (N, D) 向量。
    每个关键点 6 维：center(3) + size(3)，共 num_kps * 6 维。
    """
    if not node_indices:
        return np.zeros((0, 0), dtype=np.float32)

    idx_arr = np.array(node_indices, dtype=np.int32)
    bboxes = flat.bboxes[idx_arr]  # (N, num_kps, 6)
    centers = (bboxes[..., :3] + bboxes[..., 3:6]) * 0.5
    sizes = bboxes[..., 3:6] - bboxes[..., :3]
    features = np.concatenate([centers, sizes], axis=-1)  # (N, num_kps, 6)
    return features.reshape(len(node_indices), -1).astype(np.float32)


def cluster_leaves(
    flat: FlatOctree,
    n_clusters: int,
    random_state: int = 42,
) -> Dict[int, int]:
    """
    对叶节点进行 K-Means 聚类，返回 node_idx -> cluster_id 映射。
    """
    leaf_indices = get_leaf_nodes(flat)
    if not leaf_indices:
        return {}

    features = get_leaf_features(flat, leaf_indices)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(features)

    return {node_idx: int(label) for node_idx, label in zip(leaf_indices, labels)}


def parse_action_label(source_file: str, frame_id: Optional[str] = None) -> str:
    """
    从样本来源解析动作标签（如 Lutz、Axel、Toeloop）。

    解析优先级（从高到低）：
    1) 文件名（推荐）：例如 "Axel_10.npy" -> "Axel"
       - 适用于很多“扁平目录”数据集：.../data/Axel_10.npy
       - 也适用于 FS-Jump3D：.../Skater_A/Axel/Axel_1.npy（文件名本身就带动作前缀）
    2) FS-Jump3D 标准路径：.../npy/Skater_X/JumpType/filename.npy
    3) 最后回退：使用父目录名
    """
    # 1) 优先从文件名解析：<Action>_<Index>.<ext>
    try:
        stem = Path(source_file).stem
        if stem:
            token = stem.split("_")[0]
            # 过滤掉纯数字/空串等情况
            if token and any(ch.isalpha() for ch in token):
                return token
    except Exception:
        pass

    # 2) 尝试使用 FS-Jump3D 元数据解析（依赖路径结构）
    try:
        from npy_loader import get_npy_metadata
        meta = get_npy_metadata(source_file)
        jump_type = meta.get("jump_type", "Unknown")
        if jump_type:
            return jump_type
    except Exception:
        pass

    # 3) 回退：从目录层级猜测（兼容老数据组织方式）
    path_parts = os.path.normpath(source_file).split(os.sep)
    try:
        if "npy" in path_parts:
            npy_idx = path_parts.index("npy")
            return path_parts[npy_idx + 2] if len(path_parts) > npy_idx + 2 else "Unknown"
        if len(path_parts) >= 2:
            return path_parts[-2]
    except Exception:
        pass
    return "Unknown"


def _summarize_labels(metadata_list: List[FrameMetadata], top_n: int = 10) -> Counter:
    """统计元数据中的动作标签分布（用于诊断数据组织/标签解析是否正确）。"""
    c = Counter()
    for m in metadata_list:
        c[parse_action_label(m.source_file, getattr(m, "frame_id", None))] += 1
    return Counter(dict(c.most_common(top_n)))


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


def compute_classification_report(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict:
    """
    计算每类 precision / recall / F1，以及 macro / weighted 平均。

    参数:
        true_labels: (N,) object/string
        pred_labels: (N,) object/string
    """
    label_set = set(true_labels.tolist()) | set(pred_labels.tolist())
    labels = sorted(label_set)

    per_class: Dict[str, dict] = {}
    supports = {}
    for c in labels:
        c_true = (true_labels == c)
        c_pred = (pred_labels == c)
        tp = int(np.sum(c_true & c_pred))
        fp = int(np.sum(~c_true & c_pred))
        fn = int(np.sum(c_true & ~c_pred))
        support = int(np.sum(c_true))

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0

        per_class[str(c)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        supports[str(c)] = support

    # macro avg
    if labels:
        macro_precision = float(np.mean([per_class[c]["precision"] for c in per_class]))
        macro_recall = float(np.mean([per_class[c]["recall"] for c in per_class]))
        macro_f1 = float(np.mean([per_class[c]["f1"] for c in per_class]))
    else:
        macro_precision = macro_recall = macro_f1 = 0.0

    # weighted avg by support
    total_support = float(sum(supports.values()))
    if total_support > 0:
        weighted_precision = float(
            sum(per_class[c]["precision"] * supports[c] for c in per_class) / total_support
        )
        weighted_recall = float(
            sum(per_class[c]["recall"] * supports[c] for c in per_class) / total_support
        )
        weighted_f1 = float(
            sum(per_class[c]["f1"] * supports[c] for c in per_class) / total_support
        )
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0

    return {
        "labels": labels,
        "per_class": per_class,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
        },
        "weighted_avg": {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1": weighted_f1,
        },
    }


def build_cluster_to_action_from_metadata_a(
    tree_a: FlatOctree,
    leaf_to_cluster: Dict[int, int],
    metadata_a: List[FrameMetadata],
) -> Dict[int, str]:
    """
    用树 A 的 metadata 统计 cluster -> 动作标签的多数投票映射。

    重要：这里不使用树 B 的标签来确定映射，避免“验证集泄漏”。
    """
    if not metadata_a or not leaf_to_cluster:
        return {}

    frame_id_to_label: Dict[str, str] = {}
    for meta in metadata_a:
        frame_id_to_label[str(meta.frame_id)] = parse_action_label(meta.source_file, getattr(meta, "frame_id", None))

    # cluster_id -> Counter(label)
    cluster_label_counts: Dict[int, Counter] = defaultdict(Counter)

    # 仅遍历叶节点（leaf_to_cluster 的 key 就是叶节点）
    for leaf_idx, cid in leaf_to_cluster.items():
        # 叶节点里存的是“沿路径下沉到该叶子”的帧集合（构建时会在每层 add_frame）
        for fid in tree_a.get_frame_ids(int(leaf_idx)):
            label = frame_id_to_label.get(str(fid))
            if label is None:
                continue
            cluster_label_counts[int(cid)][label] += 1

    cluster_to_action: Dict[int, str] = {}
    for cid, counter in cluster_label_counts.items():
        if not counter:
            continue
        cluster_to_action[int(cid)] = counter.most_common(1)[0][0]

    return cluster_to_action


def validate_with_other_tree(
    tree_a: FlatOctree,
    leaf_to_cluster: Dict[int, int],
    metadata_b: List[FrameMetadata],
    *,
    cluster_to_action: Optional[Dict[int, str]] = None,
    verbose: bool = True,
) -> dict:
    """
    用树 B 的 metadata 在树 A 上验证聚类与动作标签的一致性。

    返回包含准确率、混淆矩阵等信息的字典。
    """
    if not metadata_b:
        return {"accuracy": 0.0, "total": 0, "correct": 0, "confusion": {}}

    pred_clusters: List[int] = []
    true_labels: List[str] = []

    for meta in metadata_b:
        action = parse_action_label(meta.source_file, getattr(meta, "frame_id", None))
        leaf_idx = find_leaf_node_for_query(tree_a, meta.keypoints)
        cluster_id = leaf_to_cluster.get(leaf_idx, -1)

        pred_clusters.append(cluster_id)
        true_labels.append(action)

    pred_clusters_arr = np.asarray(pred_clusters, dtype=np.int32)
    true_labels_arr = np.asarray(true_labels, dtype=object)

    # 预测动作：优先使用传入的 cluster_to_action；缺失则回退 Unknown
    if cluster_to_action is None:
        cluster_to_action = {}

    pred_actions_arr = np.asarray(
        [cluster_to_action.get(int(c), "Unknown") if int(c) >= 0 else "Unknown" for c in pred_clusters_arr],
        dtype=object,
    )

    correct = int(np.sum(pred_actions_arr == true_labels_arr))
    total = int(len(true_labels_arr))
    accuracy = correct / total if total > 0 else 0.0

    report = compute_classification_report(true_labels_arr, pred_actions_arr)

    # 混淆矩阵：true_label x pred_label
    label_set = set(true_labels_arr.tolist()) | set(pred_actions_arr.tolist())
    unique_labels = sorted(label_set)
    confusion: Dict[str, Dict[str, int]] = {t: {p: 0 for p in unique_labels} for t in unique_labels}
    for t, p in zip(true_labels_arr.tolist(), pred_actions_arr.tolist()):
        confusion[str(t)][str(p)] += 1

    missing_cluster_count = int(np.sum(pred_clusters_arr < 0))

    result = {
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "confusion": confusion,
        "cluster_to_action": cluster_to_action,
        "labels": unique_labels,
        "missing_cluster_count": missing_cluster_count,
        "report": report,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("跨树验证结果")
        print("=" * 60)
        print(f"  总帧数: {total}")
        print(f"  正确数: {correct}")
        print(f"  准确率: {accuracy:.2%}")
        print(f"  未映射到叶聚类的帧数: {missing_cluster_count}")
        print("\n聚类 -> 动作映射:")
        for cid, act in sorted(cluster_to_action.items()):
            print(f"  聚类 {cid}: {act}")
        print("\n混淆矩阵 (行=真实, 列=预测):")
        header = " " * 12 + "".join(f"{a:>10}" for a in unique_labels)
        print(header)
        for true_a in unique_labels:
            row = f"{true_a:>10}  " + "".join(
                f"{confusion[true_a].get(p, 0):>10}" for p in unique_labels
            )
            print(row)

        print("\n分类指标 (precision / recall / f1 / support):")
        # 仅展示按 label 排序后的指标
        for lbl in report["labels"]:
            m = report["per_class"][lbl]
            print(
                f"  {lbl:>10}  "
                f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
                f"support={m['support']}"
            )
        macro = report["macro_avg"]
        weighted = report["weighted_avg"]
        print(
            f"\n  macro avg    P={macro['precision']:.3f}  R={macro['recall']:.3f}  F1={macro['f1']:.3f}"
        )
        print(
            f"  weighted avg P={weighted['precision']:.3f}  R={weighted['recall']:.3f}  F1={weighted['f1']:.3f}"
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="八叉树叶节点聚类与跨树验证：对树 A 叶节点聚类，用树 B 的帧验证分类正确性"
    )
    parser.add_argument("--tree-a", required=True, help="树 A 的 .npz 文件路径")
    parser.add_argument("--metadata-a", required=True, help="树 A 的 .pkl 元数据路径")
    parser.add_argument("--metadata-b", required=True, help="树 B 的 .pkl 元数据路径（用于验证）")
    parser.add_argument("--n-clusters", type=int, default=10, help="聚类数量 K（默认 10）")
    parser.add_argument("--output", type=str, default=None, help="可选：保存聚类映射与验证报告")
    parser.add_argument("--quiet", action="store_true", help="减少输出")

    args = parser.parse_args()

    tree_path = Path(args.tree_a)
    meta_a_path = Path(args.metadata_a)
    meta_b_path = Path(args.metadata_b)

    if not tree_path.exists():
        print(f"错误: 找不到树文件 {tree_path}")
        return
    if not meta_a_path.exists():
        print(f"错误: 找不到元数据 A {meta_a_path}")
        return
    if not meta_b_path.exists():
        print(f"错误: 找不到元数据 B {meta_b_path}")
        return

    if not args.quiet:
        print("正在加载树 A...")
    tree_a = load_tree(str(tree_path), show_progress=not args.quiet)

    if not args.quiet:
        print("正在加载元数据 A...")
    metadata_a = load_metadata(str(meta_a_path))

    if not args.quiet:
        print("正在加载元数据 B（验证集）...")
    metadata_b = load_metadata(str(meta_b_path))

    if not args.quiet:
        # 诊断：检查动作标签解析是否合理（避免出现全是 "data" 之类的情况）
        dist_a = _summarize_labels(metadata_a)
        dist_b = _summarize_labels(metadata_b)
        print("\n[诊断] 元数据A动作标签分布(Top):", dict(dist_a))
        print("[诊断] 元数据B动作标签分布(Top):", dict(dist_b))

    leaf_indices = get_leaf_nodes(tree_a)
    if not args.quiet:
        print(f"\n叶节点数: {len(leaf_indices)}")

    if not args.quiet:
        print(f"正在对叶节点进行 K-Means 聚类 (K={args.n_clusters})...")
    leaf_to_cluster = cluster_leaves(tree_a, args.n_clusters)

    if not args.quiet:
        print("正在从树 A 的元数据统计 cluster -> 动作 映射（多数投票）...")
    cluster_to_action = build_cluster_to_action_from_metadata_a(tree_a, leaf_to_cluster, metadata_a)
    if not cluster_to_action and not args.quiet:
        print("警告: 未能从元数据 A 得到有效的 cluster->动作 映射（可能 frame_id 不匹配），预测将大量为 Unknown。")

    result = validate_with_other_tree(
        tree_a,
        leaf_to_cluster,
        metadata_b,
        cluster_to_action=cluster_to_action,
        verbose=not args.quiet,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # 1) 保存报告（json 或 txt，取决于后缀）
        if out_path.suffix.lower() == ".json":
            payload = dict(result)
            payload["leaf_count"] = int(len(get_leaf_nodes(tree_a)))
            payload["n_clusters"] = int(args.n_clusters)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"准确率: {result['accuracy']:.2%}\n")
                f.write(f"总帧数: {result['total']}\n")
                f.write(f"正确数: {result['correct']}\n")
                f.write(f"未映射到叶聚类的帧数: {result['missing_cluster_count']}\n\n")

                rep = result.get("report", {})
                if rep:
                    f.write("分类指标 (precision / recall / f1 / support):\n")
                    for lbl in rep.get("labels", []):
                        m = rep["per_class"][lbl]
                        f.write(
                            f"  {lbl:>10}  "
                            f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
                            f"support={m['support']}\n"
                        )
                    macro = rep.get("macro_avg", {})
                    weighted = rep.get("weighted_avg", {})
                    if macro:
                        f.write(
                            f"\n  macro avg    P={macro['precision']:.3f}  R={macro['recall']:.3f}  F1={macro['f1']:.3f}\n"
                        )
                    if weighted:
                        f.write(
                            f"  weighted avg P={weighted['precision']:.3f}  R={weighted['recall']:.3f}  F1={weighted['f1']:.3f}\n"
                        )
                    f.write("\n")

                f.write("聚类 -> 动作映射:\n")
                for cid, act in sorted(result["cluster_to_action"].items()):
                    f.write(f"  聚类 {cid}: {act}\n")
                f.write("\n混淆矩阵:\n")
                labels = result["labels"]
                f.write("true\\pred," + ",".join(labels) + "\n")
                for true_a in labels:
                    row = ",".join(str(result["confusion"][true_a].get(p, 0)) for p in labels)
                    f.write(f"{true_a},{row}\n")

        # 2) 额外保存 leaf_to_cluster 映射（npz，避免巨大 json）
        mapping_path = out_path.with_name(out_path.stem + "_leaf_to_cluster.npz")
        leaf_indices = np.asarray(sorted(leaf_to_cluster.keys()), dtype=np.int32)
        cluster_ids = np.asarray([leaf_to_cluster[int(i)] for i in leaf_indices], dtype=np.int32)
        np.savez_compressed(mapping_path, leaf_indices=leaf_indices, cluster_ids=cluster_ids)

        if not args.quiet:
            print(f"\n报告已保存至 {out_path}")
            print(f"叶节点聚类映射已保存至 {mapping_path}")


if __name__ == "__main__":
    main()
