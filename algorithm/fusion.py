# -*- coding: utf-8 -*-
"""
对多个子模型的 (score, weight) 做加权平均，并映射风险等级。
score 建议为 0~1 的概率或规则分。
"""
import math


def finite_float(v, default=0.0):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def fuse_weighted_scores(parts, cfg=None):
    """
    parts: list of (score: float, weight: float)，仅包含已启用且可用的子模型。
    cfg: 可选 dict，含 threshold_high_risk 等。
    """
    cfg = cfg or {}
    if not parts:
        return 0.0, 0

    safe = [(finite_float(v), finite_float(w)) for v, w in parts]
    safe = [(v, w) for v, w in safe if w > 0]
    if not safe:
        return 0.0, 0
    total_w = float(sum(w for _, w in safe)) or 1.0
    final_score = float(sum(v * w for v, w in safe) / total_w)

    high_risk_threshold = float(cfg.get('threshold_high_risk', 0.85))
    if final_score >= high_risk_threshold:
        level = 3
    elif final_score >= 0.65:
        level = 2
    elif final_score >= 0.4:
        level = 1
    else:
        level = 0
    return final_score, level
