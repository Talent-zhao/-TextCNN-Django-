# -*- coding: utf-8 -*-
"""模型登记：训练侧「源算法模型」；融合页同步的 name 为「融合策略」，列表「登记类别」显示「融合策略模型」。"""

SOURCE_ALGO_MODELINFO_NAME = '源算法模型'
# ModelInfo.name / version：融合策略为单条登记（model_type=fusion）
FUSION_STRATEGY_MODELINFO_NAME = '融合策略'
FUSION_STRATEGY_MODELINFO_VERSION = 'default'
FUSION_STRATEGY_MODEL_TYPE = 'fusion'
# 模型管理表「登记类别」列展示用（非 name 字段）
FUSION_STRATEGY_REGISTRY_LABEL = '融合策略模型'

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
