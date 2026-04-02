"""
基于姿态模板统计结果生成匿名规则。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

import config
from rule_classifier.template_clustering import PoseTemplate


INTERPRETABLE_FEATURES = [
    "knee_left",
    "knee_right",
    "hip_left",
    "hip_right",
    "elbow_left",
    "elbow_right",
    "torso_angle",
    "arm_span_ratio",
    "feet_distance_ratio",
    "hand_height_ratio",
    "free_leg_height",
    "free_leg_direction",
    "torso_lean_direction",
]


@dataclass
class RuleCondition:
    feature: str
    operator: str
    threshold: float


@dataclass
class TemplateRule:
    template_id: str
    conditions: List[RuleCondition]
    then_label: str
    support_leg_type: str | None
    top_features: List[Dict[str, float]]
    rule_text: str


def _resolve_feature_names(
    templates: Sequence[PoseTemplate],
    feature_source: str,
) -> List[str]:
    if feature_source == "handcrafted":
        return [feature for feature in INTERPRETABLE_FEATURES if all(feature in template.mean_feature for template in templates)]
    common = set(templates[0].mean_feature.keys()) if templates else set()
    for template in templates[1:]:
        common &= set(template.mean_feature.keys())
    return sorted(common)


def compute_template_difference_matrix(
    templates: Sequence[PoseTemplate],
    *,
    feature_source: str = "handcrafted",
) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    result: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
    for template in templates:
        result[template.template_id] = {}

    feature_names = _resolve_feature_names(templates, feature_source)
    for i in range(len(templates)):
        for j in range(i + 1, len(templates)):
            left = templates[i]
            right = templates[j]
            diffs: List[Dict[str, float]] = []
            for feature in feature_names:
                if feature not in left.mean_feature or feature not in right.mean_feature:
                    continue
                left_mean = left.mean_feature[feature]
                right_mean = right.mean_feature[feature]
                left_std = max(left.std_feature.get(feature, 0.0), 1e-6)
                right_std = max(right.std_feature.get(feature, 0.0), 1e-6)
                pooled_std = max((left_std + right_std) / 2.0, 1e-6)
                effect_size = abs(left_mean - right_mean) / pooled_std
                diffs.append(
                    {
                        "feature": feature,
                        "left_mean": float(left_mean),
                        "right_mean": float(right_mean),
                        "delta": float(left_mean - right_mean),
                        "effect_size": float(effect_size),
                    }
                )
            diffs.sort(key=lambda item: item["effect_size"], reverse=True)
            result[left.template_id][right.template_id] = diffs
            result[right.template_id][left.template_id] = [
                {
                    "feature": item["feature"],
                    "left_mean": item["right_mean"],
                    "right_mean": item["left_mean"],
                    "delta": float(-item["delta"]),
                    "effect_size": item["effect_size"],
                }
                for item in diffs
            ]
    return result


def _dominant_support_leg(template: PoseTemplate) -> str | None:
    distribution = template.categorical_distribution.get("support_leg_type", {})
    if not distribution:
        return None
    winner = max(distribution.items(), key=lambda item: item[1])
    total = sum(distribution.values())
    if total <= 0:
        return None
    if winner[1] / total < 0.7:
        return None
    return winner[0]


def _condition_from_stats(template: PoseTemplate, feature: str, others_mean: float) -> RuleCondition:
    mean_value = template.mean_feature[feature]
    std_value = max(template.std_feature.get(feature, 0.0), 1e-6)
    lower_bound = mean_value - std_value
    upper_bound = mean_value + std_value

    if lower_bound > others_mean:
        return RuleCondition(feature=feature, operator=">", threshold=float(lower_bound))
    if upper_bound < others_mean:
        return RuleCondition(feature=feature, operator="<", threshold=float(upper_bound))

    midpoint = (mean_value + others_mean) / 2.0
    operator = ">" if mean_value >= others_mean else "<"
    return RuleCondition(feature=feature, operator=operator, threshold=float(midpoint))


def generate_rules_from_templates(
    templates: Sequence[PoseTemplate],
    *,
    max_conditions: int = 3,
    feature_source: str | None = None,
) -> tuple[List[TemplateRule], Dict[str, Dict[str, List[Dict[str, float]]]]]:
    resolved_feature_source = feature_source or config.POSE_TEMPLATE_FEATURE_SOURCE
    difference_matrix = compute_template_difference_matrix(templates, feature_source=resolved_feature_source)
    rules: List[TemplateRule] = []
    is_interpretable = resolved_feature_source == "handcrafted"

    for template in templates:
        comparison_pool: List[Dict[str, float]] = []
        for other in templates:
            if other.template_id == template.template_id:
                continue
            comparison_pool.extend(difference_matrix[template.template_id][other.template_id][:5])

        best_by_feature: Dict[str, Dict[str, float]] = {}
        for item in comparison_pool:
            current = best_by_feature.get(item["feature"])
            if current is None or item["effect_size"] > current["effect_size"]:
                best_by_feature[item["feature"]] = item

        top_features = sorted(best_by_feature.values(), key=lambda x: x["effect_size"], reverse=True)[:max_conditions]

        conditions: List[RuleCondition] = []
        support_leg_type = _dominant_support_leg(template) if is_interpretable else None
        if is_interpretable:
            for item in top_features:
                feature = item["feature"]
                other_means = [
                    other.mean_feature[feature]
                    for other in templates
                    if other.template_id != template.template_id and feature in other.mean_feature
                ]
                if not other_means:
                    continue
                conditions.append(_condition_from_stats(template, feature, float(np.mean(other_means))))

            if support_leg_type:
                top_features.append(
                    {
                        "feature": "support_leg_type",
                        "left_mean": 0.0,
                        "right_mean": 0.0,
                        "delta": 0.0,
                        "effect_size": 1.0,
                    }
                )

            condition_lines = [f"{cond.feature} {cond.operator} {cond.threshold:.4f}" for cond in conditions]
            if support_leg_type:
                condition_lines.append(f"support_leg_type == {support_leg_type}")

            if not condition_lines:
                rule_text = f"THEN {template.template_id}"
            else:
                rule_text = "IF\n" + "\nAND ".join(condition_lines) + f"\nTHEN {template.template_id}"
        else:
            top_desc = "\n".join(
                f"- {item['feature']} (effect_size={float(item['effect_size']):.4f}, delta={float(item['delta']):.4f})"
                for item in top_features[:max_conditions]
            ) or "- 无足够差异特征"
            rule_text = (
                f"EXPERIMENTAL_FEATURE_SOURCE: {resolved_feature_source}\n"
                "当前特征源默认只输出实验性分析，不生成正式可解释规则。\n"
                "Top distinguishing features:\n"
                f"{top_desc}\n"
                f"THEN {template.template_id}"
            )

        rules.append(
            TemplateRule(
                template_id=template.template_id,
                conditions=conditions,
                then_label=template.template_id,
                support_leg_type=support_leg_type,
                top_features=top_features,
                rule_text=rule_text,
            )
        )

    return rules, difference_matrix


def serialize_rules(rules: Sequence[TemplateRule]) -> List[Dict[str, object]]:
    payload: List[Dict[str, object]] = []
    for rule in rules:
        payload.append(
            {
                "template_id": rule.template_id,
                "conditions": [
                    {
                        "feature": cond.feature,
                        "operator": cond.operator,
                        "threshold": cond.threshold,
                    }
                    for cond in rule.conditions
                ],
                "support_leg_type": rule.support_leg_type,
                "then_label": rule.then_label,
                "top_features": rule.top_features,
                "rule_text": rule.rule_text,
            }
        )
    return payload

