"""
批量构建姿态模板、规则草案与质量报告。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config
from rule_classifier.rule_generator import generate_rules_from_templates, serialize_rules
from rule_classifier.template_clustering import run_clustering_pipeline, save_clustering_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="构建姿态模板与规则草案")
    parser.add_argument("--data-dir", default=config.FS_JUMP3D_DATA_DIR, help="输入 NPY 数据目录")
    parser.add_argument("--output-dir", default="output/pose_templates", help="输出目录")
    parser.add_argument("--sample-stride", type=int, default=5, help="采样步长")
    parser.add_argument("--max-files", type=int, default=None, help="最多处理多少个文件")
    parser.add_argument("--max-frames-per-file", type=int, default=None, help="每个文件最多处理多少帧")
    parser.add_argument("--feature-source", choices=["handcrafted", "coordinates", "octree_node"], default=config.POSE_TEMPLATE_FEATURE_SOURCE, help="姿态模板特征源")
    parser.add_argument("--coordinate-mode", default=config.POSE_TEMPLATE_COORDINATE_MODE, help="坐标特征模式")
    parser.add_argument("--octree-node-mode", default=config.POSE_TEMPLATE_OCTREE_NODE_MODE, help="八叉树实验特征模式")
    parser.add_argument("--pca-components", type=float, default=config.POSE_TEMPLATE_PCA_COMPONENTS, help="PCA 保留方差比例")
    parser.add_argument("--dbscan-eps", type=float, default=1.2, help="DBSCAN eps")
    parser.add_argument("--dbscan-min-samples", type=int, default=20, help="DBSCAN min_samples")
    parser.add_argument("--kmedoids-max-k", type=int, default=6, help="K-Medoids 最大 K")
    parser.add_argument("--kmedoids-target-cluster-size", type=int, default=120, help="K-Medoids 目标簇大小")
    parser.set_defaults(use_pca=None, use_zscore=None)
    parser.add_argument("--use-pca", dest="use_pca", action="store_true", help="强制启用 PCA")
    parser.add_argument("--disable-pca", dest="use_pca", action="store_false", help="强制关闭 PCA")
    parser.add_argument("--use-zscore", dest="use_zscore", action="store_true", help="强制启用 Z-score")
    parser.add_argument("--disable-zscore", dest="use_zscore", action="store_false", help="强制关闭 Z-score")
    args = parser.parse_args()

    artifacts = run_clustering_pipeline(
        args.data_dir,
        sample_stride=args.sample_stride,
        max_files=args.max_files,
        max_frames_per_file=args.max_frames_per_file,
        feature_source=args.feature_source,
        coordinate_mode=args.coordinate_mode,
        octree_node_mode=args.octree_node_mode,
        use_zscore=args.use_zscore,
        use_pca=args.use_pca,
        pca_components=args.pca_components,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        kmedoids_max_k=args.kmedoids_max_k,
        kmedoids_target_cluster_size=args.kmedoids_target_cluster_size,
    )
    saved_paths = save_clustering_artifacts(artifacts, args.output_dir)

    rules, difference_matrix = generate_rules_from_templates(
        artifacts.templates,
        feature_source=artifacts.feature_source,
    )
    rules_path = Path(args.output_dir) / "anonymous_rules.json"
    diff_path = Path(args.output_dir) / "template_difference_analysis.json"
    cluster_names_path = Path(args.output_dir) / "cluster_names.json"

    rules_path.write_text(
        json.dumps({"rules": serialize_rules(rules)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    diff_path.write_text(json.dumps(difference_matrix, ensure_ascii=False, indent=2), encoding="utf-8")
    if not cluster_names_path.exists():
        cluster_names_path.write_text(json.dumps({}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("姿态模板构建完成")
    print(f"特征源: {artifacts.feature_source}")
    if artifacts.feature_source == "coordinates":
        print(f"坐标模式: {artifacts.coordinate_mode}")
    if artifacts.feature_source == "octree_node":
        print(f"八叉树模式: {artifacts.octree_node_mode}")
    print(f"使用 Z-score: {artifacts.use_zscore}")
    print(f"使用 PCA: {artifacts.use_pca}")
    print(f"样本数: {artifacts.metadata_summary['sample_count']}")
    print(f"DBSCAN 粗簇数: {artifacts.metadata_summary['dbscan_cluster_count']}")
    print(f"最终模板数: {artifacts.metadata_summary['final_cluster_count']}")
    print(f"噪声样本数: {artifacts.metadata_summary['noise_count']}")
    print(f"PCA 解释方差: {artifacts.metadata_summary['pca_explained_variance_ratio']:.4f}")
    print("-" * 72)
    print(f"模板文件: {saved_paths['templates']}")
    print(f"分配文件: {saved_paths['assignments']}")
    print(f"质量报告: {saved_paths['quality_report']}")
    print(f"距离矩阵: {saved_paths['distance_matrix']}")
    print(f"匿名规则: {rules_path}")
    print(f"差异分析: {diff_path}")
    print(f"命名映射: {cluster_names_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()

