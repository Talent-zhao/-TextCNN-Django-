# -*- coding: utf-8 -*-
"""模型登记：训练侧「源算法模型」；融合页同步的 name 为「融合策略」，列表「登记类别」显示「融合策略模型」。"""

import re

SOURCE_ALGO_MODELINFO_NAME = '源算法模型'
# ModelInfo.name / version：融合策略为单条登记（model_type=fusion）
FUSION_STRATEGY_MODELINFO_NAME = '融合策略'
FUSION_STRATEGY_MODELINFO_VERSION = 'default'
FUSION_STRATEGY_MODEL_TYPE = 'fusion'
# 模型管理表「登记类别」列展示用（非 name 字段）
FUSION_STRATEGY_REGISTRY_LABEL = '融合策略模型'
# 模型管理「新增融合策略登记」自动名称前缀：策略模型01、策略模型02 …
FUSION_STRATEGY_AUTO_NAME_PREFIX = '策略模型'

LEGACY_SOURCE_ALGO_NAMES = frozenset({'训练自动评估'})
LEGACY_FUSION_STRATEGY_NAMES = frozenset(
    {'融合配置同步', '模型融合策略模型'}
)


def source_algo_modelinfo_name_set():
    return {SOURCE_ALGO_MODELINFO_NAME} | LEGACY_SOURCE_ALGO_NAMES


def fusion_strategy_modelinfo_name_set():
    return {FUSION_STRATEGY_MODELINFO_NAME} | LEGACY_FUSION_STRATEGY_NAMES


SOURCE_ALGO_TYPES = frozenset({'svm', 'knn', 'rf', 'dt', 'lr', 'textcnn', 'textrcnn'})


def is_canonical_fusion_registry_row(mi):
    """融合策略规范登记：类型为 fusion 且版本为 default（名称可自定义，如「融合策略002」）。"""
    return (
        getattr(mi, 'model_type', None) == FUSION_STRATEGY_MODEL_TYPE
        and (mi.version or '').strip() == FUSION_STRATEGY_MODELINFO_VERSION
    )


def is_primary_fusion_system_sync_row(mi):
    """
    与「预测融合设置」全局快照绑定的唯一登记：名称须为「融合策略」且 version=default。
    其他名称的融合策略（如 test012）仅使用各自 metrics_json / config_json，不会被 sync 整表覆盖。
    """
    if not is_canonical_fusion_registry_row(mi):
        return False
    return (mi.name or '').strip() == FUSION_STRATEGY_MODELINFO_NAME


def next_autonamed_fusion_strategy_model_name():
    """
    下一条自动命名的融合策略登记名称：在已有「策略模型」+ 纯数字的名称中取最大序号 +1，
    至少两位宽（01、02…；满两位后继续 99、100）。
    手工命名的融合登记不参与占用序号。
    """
    from app.models import ModelInfo

    prefix = FUSION_STRATEGY_AUTO_NAME_PREFIX
    pattern = re.compile(r'^' + re.escape(prefix) + r'(\d+)$')
    max_n = 0
    for name in ModelInfo.objects.filter(model_type=FUSION_STRATEGY_MODEL_TYPE).values_list(
        'name', flat=True
    ):
        m = pattern.match((name or '').strip())
        if m:
            max_n = max(max_n, int(m.group(1)))
    return '{}{:02d}'.format(prefix, max_n + 1)


def registry_display_label(mi, eval_record_name, eval_record_version):
    """
    模型管理列表「登记类别」列：不修改数据库，仅展示文案。
    仅依据 name / model_type 等稳定字段，不因说明文案等编辑而漂移。
    """
    n = (mi.name or '').strip()
    ver = (mi.version or '').strip()
    mt = getattr(mi, 'model_type', None)
    if n == eval_record_name and ver == eval_record_version:
        return '系统占位'
    if is_canonical_fusion_registry_row(mi):
        return FUSION_STRATEGY_REGISTRY_LABEL
    if (n in source_algo_modelinfo_name_set() or n in LEGACY_SOURCE_ALGO_NAMES) and mt in SOURCE_ALGO_TYPES:
        return SOURCE_ALGO_MODELINFO_NAME
    if n == 'svm_baseline' and ver == 'v1':
        return '默认基线'
    return '其他登记'
