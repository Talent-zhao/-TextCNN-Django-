# -*- coding: utf-8 -*-
"""模型授权方案：按「模型登记」ID 或（兼容旧数据）按 model_type 限制。"""

import json


def parse_plan_id_allow(json_str):
    """
    None：未配置 ID 限制（字段为空）。
    set：已配置；空集表示显式不允许任何登记（应无可用模型）。
    """
    raw = (json_str or '').strip()
    if not raw:
        return None
    try:
        arr = json.loads(raw)
        if not isinstance(arr, list):
            return set()
        out = set()
        for x in arr:
            try:
                out.add(int(x))
            except (TypeError, ValueError):
                pass
        return out
    except (TypeError, ValueError, json.JSONDecodeError):
        return set()


def parse_plan_type_allow(json_str):
    raw = (json_str or '').strip()
    if not raw:
        return None
    try:
        arr = json.loads(raw)
        if not isinstance(arr, list):
            return set()
        return {str(x) for x in arr}
    except (TypeError, ValueError, json.JSONDecodeError):
        return set()


def model_info_passes_plan(model_info, plan):
    if model_info is None or plan is None:
        return False
    id_allow = parse_plan_id_allow(plan.allowed_model_ids_json)
    t_allow = parse_plan_type_allow(plan.allowed_model_types_json)
    if id_allow is not None:
        if len(id_allow) == 0:
            return False
        return model_info.id in id_allow
    if t_allow is not None:
        if len(t_allow) == 0:
            return False
        return model_info.model_type in t_allow
    return True


def plan_model_allowsets(plan):
    """与旧 views 一致，供需同时拿到两套集合的代码使用。"""
    return parse_plan_type_allow(plan.allowed_model_types_json), parse_plan_id_allow(plan.allowed_model_ids_json)
