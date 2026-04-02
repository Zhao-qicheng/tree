"""
姿态模板聚类流水线。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score

import config
from npy_loader import get_npy_metadata, load_all_npy_files, load_npy_file
from rule_classifier.pose_features import PoseFeatureBundle, extract_pose_features_by_source, preprocess_frame


@dataclass
class PoseSample:
    sample_id: int
    source_file: str
    frame_index: int
    file_name: str
    skater: str
    jump_type: str
    processed_frame: np.ndarray
    feature_bundle: PoseFeatureBundle


@dataclass
class PoseTemplate:
    template_id: str
    template_index: int
    medoid_sample_id: int
    medoid_pose: list[list[float]]
    cluster_size: int
    cluster_radius: float
    mean_feature: Dict[str, float]
    std_feature: Dict[str, float]
    categorical_distribution: Dict[str, Dict[str, int]]
    member_frames: List[Dict[str, object]]
    quality_flags: List[str]


@dataclass
class ClusteringArtifacts:
    samples: List[PoseSample]
    feature_source: str
    coordinate_mode: str
    octree_node_mode: str
    use_zscore: bool
    use_pca: bool
    vector_feature_names: List[str]
    report_feature_names: List[str]
    categorical_feature_names: List[str]
    raw_feature_matrix: np.ndarray
    feature_matrix: np.ndarray
    zscore_matrix: np.ndarray | None
    zscore_mean: np.ndarray | None
    zscore_std: np.ndarray | None
    cluster_matrix: np.ndarray
    pca_matrix: np.ndarray | None
    dbscan_labels: np.ndarray
    final_labels: np.ndarray
    templates: List[PoseTemplate]
    template_distance_matrix: np.ndarray
    split_candidates: List[Dict[str, object]]
    merge_candidates: List[Dict[str, object]]
    metadata_summary: Dict[str, object]


def _require_kmedoids():
    try:
        from sklearn_extra.cluster import KMedoids
        return KMedoids
    except Exception:
        return None


def _kmedoids_fallback_labels(
    matrix: np.ndarray,
    n_clusters: int,
    *,
    max_iter: int = 100,
    random_state: int = 42,
) -> np.ndarray:
    if n_clusters <= 1:
        return np.zeros(matrix.shape[0], dtype=np.int32)
    if n_clusters >= matrix.shape[0]:
        return np.arange(matrix.shape[0], dtype=np.int32)

    rng = np.random.default_rng(random_state)
    distances = pairwise_distances(matrix, metric="euclidean")
    medoids = [int(rng.integers(matrix.shape[0]))]

    while len(medoids) < n_clusters:
        min_dist = distances[:, medoids].min(axis=1)
        next_idx = int(np.argmax(min_dist))
        if next_idx in medoids:
            remaining = [idx for idx in range(matrix.shape[0]) if idx not in medoids]
            next_idx = int(remaining[0])
        medoids.append(next_idx)

    medoids = np.array(medoids, dtype=np.int32)
    labels = np.zeros(matrix.shape[0], dtype=np.int32)
    for _ in range(max_iter):
        labels = np.argmin(distances[:, medoids], axis=1).astype(np.int32)
        new_medoids = medoids.copy()
        changed = False
        for cluster_idx in range(n_clusters):
            members = np.flatnonzero(labels == cluster_idx)
            if len(members) == 0:
                candidates = [idx for idx in range(matrix.shape[0]) if idx not in new_medoids.tolist()]
                if candidates:
                    new_medoids[cluster_idx] = int(candidates[0])
                    changed = True
                continue
            member_dist = distances[np.ix_(members, members)]
            best_local = int(members[np.argmin(member_dist.sum(axis=1))])
            if new_medoids[cluster_idx] != best_local:
                new_medoids[cluster_idx] = best_local
                changed = True
        medoids = new_medoids
        if not changed:
            break
    return np.argmin(distances[:, medoids], axis=1).astype(np.int32)


def _fit_kmedoids_labels(matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    KMedoids = _require_kmedoids()
    if KMedoids is not None:
        model = KMedoids(n_clusters=n_clusters, metric="euclidean", init="k-medoids++", random_state=42)
        return model.fit_predict(matrix)
    return _kmedoids_fallback_labels(matrix, n_clusters=n_clusters)


def load_pose_samples(
    data_dir: str,
    *,
    sample_stride: int = 1,
    max_files: int | None = None,
    max_frames_per_file: int | None = None,
    feature_source: str = "handcrafted",
    coordinate_mode: str = "flattened",
    octree_node_mode: str = "path_encoding",
) -> List[PoseSample]:
    if sample_stride <= 0:
        raise ValueError("sample_stride 必须大于 0")

    npy_files = load_all_npy_files(data_dir)
    if max_files is not None:
        npy_files = npy_files[:max_files]

    samples: List[PoseSample] = []
    sample_id = 0
    for npy_file in npy_files:
        frames = load_npy_file(npy_file)
        file_meta = get_npy_metadata(npy_file)
        frame_indices = range(0, frames.shape[0], sample_stride)
        if max_frames_per_file is not None:
            frame_indices = list(frame_indices)[:max_frames_per_file]

        for frame_index in frame_indices:
            processed = preprocess_frame(frames[frame_index], align=True, normalize=True)
            bundle = extract_pose_features_by_source(
                feature_source,
                processed,
                coordinate_mode=coordinate_mode,
                octree_mode=octree_node_mode,
            )
            samples.append(
                PoseSample(
                    sample_id=sample_id,
                    source_file=os.path.abspath(npy_file),
                    frame_index=int(frame_index),
                    file_name=Path(npy_file).name,
                    skater=file_meta.get("skater", "Unknown"),
                    jump_type=file_meta.get("jump_type", "Unknown"),
                    processed_frame=processed,
                    feature_bundle=bundle,
                )
            )
            sample_id += 1
    return samples


def _stack_feature_matrix(samples: Sequence[PoseSample]) -> tuple[np.ndarray, List[str], List[str], List[str], np.ndarray]:
    if not samples:
        raise ValueError("没有可用样本，无法构建特征矩阵")

    vector_feature_names = samples[0].feature_bundle.vector_feature_names
    report_feature_names = list(samples[0].feature_bundle.report_features.keys())
    categorical_feature_names = list(samples[0].feature_bundle.categorical_features.keys())

    vector_matrix = np.vstack([sample.feature_bundle.vector for sample in samples]).astype(np.float64)
    report_matrix = np.vstack(
        [
            np.array([sample.feature_bundle.report_features[name] for name in report_feature_names], dtype=np.float64)
            for sample in samples
        ]
    )
    return vector_matrix, vector_feature_names, report_feature_names, categorical_feature_names, report_matrix


def zscore_normalize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = (matrix - mean) / std
    return normalized, mean, std


def apply_pca(matrix: np.ndarray, *, n_components: float | int = 0.95) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=n_components, svd_solver="full", random_state=42)
    transformed = pca.fit_transform(matrix)
    return transformed, pca


def run_dbscan(matrix: np.ndarray, *, eps: float = 1.2, min_samples: int = 20) -> np.ndarray:
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    return model.fit_predict(matrix)


def _estimate_kmedoids_k(cluster_size: int, min_cluster_size: int, max_k: int, target_cluster_size: int) -> int:
    if cluster_size < min_cluster_size * 2:
        return 1
    candidate = int(round(cluster_size / max(target_cluster_size, 1)))
    return max(1, min(max_k, candidate))


def refine_clusters_with_kmedoids(
    pca_matrix: np.ndarray,
    dbscan_labels: np.ndarray,
    *,
    min_cluster_size: int = 20,
    max_k: int = 6,
    target_cluster_size: int = 120,
) -> np.ndarray:
    final_labels = np.full_like(dbscan_labels, -1)
    next_label = 0

    unique_labels = sorted(label for label in np.unique(dbscan_labels) if label != -1)
    for coarse_label in unique_labels:
        member_idx = np.flatnonzero(dbscan_labels == coarse_label)
        cluster_size = len(member_idx)
        if cluster_size == 0:
            continue

        cluster_matrix = pca_matrix[member_idx]
        base_k = _estimate_kmedoids_k(cluster_size, min_cluster_size, max_k, target_cluster_size)
        if base_k <= 1:
            final_labels[member_idx] = next_label
            next_label += 1
            continue

        best_score = float("-inf")
        best_labels: np.ndarray | None = None
        candidate_upper = min(base_k, max(2, cluster_size // max(min_cluster_size, 2)))
        for k in range(2, candidate_upper + 1):
            if k >= cluster_size:
                break
            labels = _fit_kmedoids_labels(cluster_matrix, n_clusters=k)
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(cluster_matrix, labels)
            if score > best_score:
                best_score = score
                best_labels = labels

        if best_labels is None:
            final_labels[member_idx] = next_label
            next_label += 1
            continue

        for local_label in sorted(np.unique(best_labels)):
            local_member_idx = member_idx[best_labels == local_label]
            final_labels[local_member_idx] = next_label
            next_label += 1

    return final_labels


def _medoid_index(feature_matrix: np.ndarray, member_idx: np.ndarray) -> int:
    if len(member_idx) == 1:
        return int(member_idx[0])
    subset = feature_matrix[member_idx]
    distances = pairwise_distances(subset, metric="euclidean")
    medoid_local = int(np.argmin(distances.sum(axis=1)))
    return int(member_idx[medoid_local])


def _categorical_distribution(samples: Sequence[PoseSample], member_idx: np.ndarray) -> Dict[str, Dict[str, int]]:
    results: Dict[str, Dict[str, int]] = {}
    for idx in member_idx:
        for key, value in samples[idx].feature_bundle.categorical_features.items():
            results.setdefault(key, {})
            results[key][value] = results[key].get(value, 0) + 1
    return results


def build_pose_templates(
    samples: Sequence[PoseSample],
    report_feature_names: Sequence[str],
    feature_matrix: np.ndarray,
    final_labels: np.ndarray,
    *,
    split_radius_threshold: float | None = None,
) -> List[PoseTemplate]:
    templates: List[PoseTemplate] = []
    valid_labels = sorted(label for label in np.unique(final_labels) if label != -1)

    radii: List[float] = []
    template_records = []
    for template_index, label in enumerate(valid_labels):
        member_idx = np.flatnonzero(final_labels == label)
        medoid_idx = _medoid_index(feature_matrix, member_idx)
        medoid_vector = feature_matrix[medoid_idx]
        cluster_matrix = feature_matrix[member_idx]
        cluster_distances = np.linalg.norm(cluster_matrix - medoid_vector, axis=1)
        cluster_radius = float(cluster_distances.max()) if len(cluster_distances) else 0.0
        radii.append(cluster_radius)

        report_matrix = np.vstack(
            [
                np.array([samples[idx].feature_bundle.report_features[name] for name in report_feature_names], dtype=np.float64)
                for idx in member_idx
            ]
        )
        mean_feature = {
            name: float(report_matrix[:, col_idx].mean())
            for col_idx, name in enumerate(report_feature_names)
        }
        std_feature = {
            name: float(report_matrix[:, col_idx].std())
            for col_idx, name in enumerate(report_feature_names)
        }
        member_frames = [
            {
                "sample_id": int(samples[idx].sample_id),
                "source_file": samples[idx].source_file,
                "file_name": samples[idx].file_name,
                "frame_index": int(samples[idx].frame_index),
                "skater": samples[idx].skater,
                "jump_type": samples[idx].jump_type,
            }
            for idx in member_idx
        ]
        template_records.append(
            (
                template_index,
                medoid_idx,
                cluster_radius,
                mean_feature,
                std_feature,
                _categorical_distribution(samples, member_idx),
                member_frames,
                [],
            )
        )

    auto_threshold = float(np.median(radii) * 1.5) if radii else 0.0
    radius_threshold = split_radius_threshold if split_radius_threshold is not None else auto_threshold

    for (
        template_index,
        medoid_idx,
        cluster_radius,
        mean_feature,
        std_feature,
        categorical_distribution,
        member_frames,
        quality_flags,
    ) in template_records:
        if cluster_radius > radius_threshold and radius_threshold > 0:
            quality_flags.append("needs_split")

        templates.append(
            PoseTemplate(
                template_id=f"Cluster_{template_index:02d}",
                template_index=int(template_index),
                medoid_sample_id=int(samples[medoid_idx].sample_id),
                medoid_pose=samples[medoid_idx].processed_frame.tolist(),
                cluster_size=int(len(member_frames)),
                cluster_radius=float(cluster_radius),
                mean_feature=mean_feature,
                std_feature=std_feature,
                categorical_distribution=categorical_distribution,
                member_frames=member_frames,
                quality_flags=quality_flags,
            )
        )
    return templates


def compute_template_distance_matrix(
    templates: Sequence[PoseTemplate],
    report_feature_names: Sequence[str],
) -> np.ndarray:
    if not templates:
        return np.zeros((0, 0), dtype=np.float64)
    matrix = np.vstack(
        [
            np.array([template.mean_feature[name] for name in report_feature_names], dtype=np.float64)
            for template in templates
        ]
    )
    return pairwise_distances(matrix, metric="euclidean")


def analyze_template_quality(
    templates: Sequence[PoseTemplate],
    distance_matrix: np.ndarray,
    *,
    merge_distance_threshold: float | None = None,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    split_candidates = [
        {
            "template_id": template.template_id,
            "cluster_radius": template.cluster_radius,
            "cluster_size": template.cluster_size,
            "quality_flags": list(template.quality_flags),
        }
        for template in templates
        if "needs_split" in template.quality_flags
    ]

    merge_candidates: List[Dict[str, object]] = []
    if distance_matrix.size == 0 or len(templates) < 2:
        return split_candidates, merge_candidates

    nonzero = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    if len(nonzero) == 0:
        return split_candidates, merge_candidates

    threshold = merge_distance_threshold
    if threshold is None:
        threshold = float(np.percentile(nonzero, 20))

    for i in range(len(templates)):
        for j in range(i + 1, len(templates)):
            if distance_matrix[i, j] < threshold:
                merge_candidates.append(
                    {
                        "template_a": templates[i].template_id,
                        "template_b": templates[j].template_id,
                        "distance": float(distance_matrix[i, j]),
                        "quality_flag": "needs_merge",
                    }
                )
    return split_candidates, merge_candidates


def run_clustering_pipeline(
    data_dir: str,
    *,
    sample_stride: int = 1,
    max_files: int | None = None,
    max_frames_per_file: int | None = None,
    feature_source: str | None = None,
    coordinate_mode: str | None = None,
    octree_node_mode: str | None = None,
    use_zscore: bool | None = None,
    use_pca: bool | None = None,
    pca_components: float | int | None = None,
    dbscan_eps: float = 1.2,
    dbscan_min_samples: int = 20,
    kmedoids_max_k: int = 6,
    kmedoids_target_cluster_size: int = 120,
    split_radius_threshold: float | None = None,
    merge_distance_threshold: float | None = None,
) -> ClusteringArtifacts:
    resolved_feature_source = feature_source or config.POSE_TEMPLATE_FEATURE_SOURCE
    resolved_coordinate_mode = coordinate_mode or config.POSE_TEMPLATE_COORDINATE_MODE
    resolved_octree_node_mode = octree_node_mode or config.POSE_TEMPLATE_OCTREE_NODE_MODE
    resolved_use_zscore = config.POSE_TEMPLATE_USE_ZSCORE if use_zscore is None else use_zscore
    if use_pca is None:
        if resolved_feature_source == "octree_node":
            resolved_use_pca = config.POSE_TEMPLATE_OCTREE_USE_PCA
        else:
            resolved_use_pca = config.POSE_TEMPLATE_USE_PCA
    else:
        resolved_use_pca = use_pca
    resolved_pca_components = pca_components if pca_components is not None else config.POSE_TEMPLATE_PCA_COMPONENTS

    samples = load_pose_samples(
        data_dir,
        sample_stride=sample_stride,
        max_files=max_files,
        max_frames_per_file=max_frames_per_file,
        feature_source=resolved_feature_source,
        coordinate_mode=resolved_coordinate_mode,
        octree_node_mode=resolved_octree_node_mode,
    )
    vector_matrix, vector_feature_names, report_feature_names, categorical_feature_names, report_matrix = _stack_feature_matrix(samples)
    feature_matrix = vector_matrix
    zscore_matrix: np.ndarray | None = None
    zscore_mean: np.ndarray | None = None
    zscore_std: np.ndarray | None = None
    cluster_matrix = feature_matrix
    pca_matrix: np.ndarray | None = None
    pca_model: PCA | None = None

    if resolved_use_zscore:
        zscore_matrix, zscore_mean, zscore_std = zscore_normalize(feature_matrix)
        cluster_matrix = zscore_matrix

    if resolved_use_pca:
        pca_matrix, pca_model = apply_pca(cluster_matrix, n_components=resolved_pca_components)
        cluster_matrix = pca_matrix

    dbscan_labels = run_dbscan(cluster_matrix, eps=dbscan_eps, min_samples=dbscan_min_samples)
    final_labels = refine_clusters_with_kmedoids(
        cluster_matrix,
        dbscan_labels,
        min_cluster_size=dbscan_min_samples,
        max_k=kmedoids_max_k,
        target_cluster_size=kmedoids_target_cluster_size,
    )
    templates = build_pose_templates(
        samples,
        report_feature_names,
        cluster_matrix,
        final_labels,
        split_radius_threshold=split_radius_threshold,
    )
    distance_matrix = compute_template_distance_matrix(templates, report_feature_names)
    split_candidates, merge_candidates = analyze_template_quality(
        templates,
        distance_matrix,
        merge_distance_threshold=merge_distance_threshold,
    )

    metadata_summary = {
        "feature_source": resolved_feature_source,
        "coordinate_mode": resolved_coordinate_mode,
        "octree_node_mode": resolved_octree_node_mode,
        "use_zscore": bool(resolved_use_zscore),
        "use_pca": bool(resolved_use_pca),
        "sample_count": len(samples),
        "dbscan_cluster_count": int(len(set(dbscan_labels.tolist())) - (1 if -1 in dbscan_labels else 0)),
        "final_cluster_count": int(len(set(final_labels.tolist())) - (1 if -1 in final_labels else 0)),
        "noise_count": int(np.sum(dbscan_labels == -1)),
        "cluster_input_dim": int(cluster_matrix.shape[1]) if cluster_matrix.ndim == 2 else 0,
        "pca_components": int(pca_matrix.shape[1]) if pca_matrix is not None and pca_matrix.ndim == 2 else 0,
        "pca_explained_variance_ratio": float(pca_model.explained_variance_ratio_.sum()) if pca_model is not None else 0.0,
    }

    return ClusteringArtifacts(
        samples=list(samples),
        feature_source=resolved_feature_source,
        coordinate_mode=resolved_coordinate_mode,
        octree_node_mode=resolved_octree_node_mode,
        use_zscore=bool(resolved_use_zscore),
        use_pca=bool(resolved_use_pca),
        vector_feature_names=vector_feature_names,
        report_feature_names=report_feature_names,
        categorical_feature_names=categorical_feature_names,
        raw_feature_matrix=report_matrix,
        feature_matrix=feature_matrix,
        zscore_matrix=zscore_matrix,
        zscore_mean=zscore_mean,
        zscore_std=zscore_std,
        cluster_matrix=cluster_matrix,
        pca_matrix=pca_matrix,
        dbscan_labels=dbscan_labels,
        final_labels=final_labels,
        templates=templates,
        template_distance_matrix=distance_matrix,
        split_candidates=split_candidates,
        merge_candidates=merge_candidates,
        metadata_summary=metadata_summary,
    )


def save_clustering_artifacts(artifacts: ClusteringArtifacts, output_dir: str) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    output_root = Path(output_dir)

    assignments_path = output_root / "cluster_assignments.csv"
    templates_path = output_root / "pose_templates.json"
    matrix_path = output_root / "template_distance_matrix.json"
    summary_path = output_root / "template_quality_report.json"

    with assignments_path.open("w", encoding="utf-8") as f:
        f.write("sample_id,source_file,file_name,frame_index,skater,jump_type,dbscan_label,template_id\n")
        for sample, db_label, final_label in zip(artifacts.samples, artifacts.dbscan_labels, artifacts.final_labels):
            template_id = "" if final_label == -1 else f"Cluster_{int(final_label):02d}"
            f.write(
                f"{sample.sample_id},"
                f"\"{sample.source_file}\","
                f"\"{sample.file_name}\","
                f"{sample.frame_index},"
                f"\"{sample.skater}\","
                f"\"{sample.jump_type}\","
                f"{int(db_label)},"
                f"{template_id}\n"
            )

    template_payload = {
        "metadata_summary": artifacts.metadata_summary,
        "vector_feature_names": artifacts.vector_feature_names,
        "report_feature_names": artifacts.report_feature_names,
        "categorical_feature_names": artifacts.categorical_feature_names,
        "zscore_mean": artifacts.zscore_mean.tolist() if artifacts.zscore_mean is not None else None,
        "zscore_std": artifacts.zscore_std.tolist() if artifacts.zscore_std is not None else None,
        "templates": [asdict(template) for template in artifacts.templates],
    }
    templates_path.write_text(json.dumps(template_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    matrix_path.write_text(
        json.dumps(
            {
                "template_ids": [template.template_id for template in artifacts.templates],
                "distance_matrix": artifacts.template_distance_matrix.tolist(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_payload = {
        "metadata_summary": artifacts.metadata_summary,
        "split_candidates": artifacts.split_candidates,
        "merge_candidates": artifacts.merge_candidates,
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "assignments": str(assignments_path),
        "templates": str(templates_path),
        "distance_matrix": str(matrix_path),
        "quality_report": str(summary_path),
    }

