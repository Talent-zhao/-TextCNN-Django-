import csv
import json
import os
from datetime import datetime
from urllib.parse import urlencode

from django.conf import settings
from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views import View
from django.utils import timezone
from django.utils.decorators import method_decorator

from app.models import (
    AlgorithmExperimentRecord,
    ExportLog,
    FusionConfigPreset,
    ModelEvaluation,
    ModelInfo,
    ModelSelfCheckRecord,
    SystemConfig,
    User,
)
from app.registry_names import (
    FUSION_STRATEGY_MODELINFO_NAME,
    FUSION_STRATEGY_MODELINFO_VERSION,
    FUSION_STRATEGY_MODEL_TYPE,
    SOURCE_ALGO_TYPES,
    is_canonical_fusion_registry_row,
    is_primary_fusion_system_sync_row,
    registry_display_label,
    source_algo_modelinfo_name_set,
)
from app.services.fusion_strategy_report import get_cached_fusion_strategy_report, report_to_table_row
from app.services.model_health_service import ModelHealthService
from app.services.predict_service import PredictService
from app.userViews import check_admin_access, get_admin_panel_user

# 训练脚本写 ModelEvaluation 时若库中无同类型模型，会用到占位 ModelInfo；同时保证「评估结果」筛选下拉不为空
EVAL_RECORD_NAME = '线下评估记录'
EVAL_RECORD_VERSION = 'default'


def _ensure_evaluation_modelinfo():
    ModelInfo.objects.get_or_create(
        name=EVAL_RECORD_NAME,
        version=EVAL_RECORD_VERSION,
        defaults={
            'model_type': 'fusion',
            'status': 'ready',
            'is_active': False,
            'file_path': '',
            'vectorizer_path': '',
            'description': '系统占位：供训练自动写评估等使用；可在模型管理中维护正式条目。',
            'config_json': '{}',
            'metrics_json': '{}',
        },
    )


ALGO_COMPARE_DEFS = (
    ('svm', 'SVM'),
    ('knn', 'KNN'),
    ('dt', '决策树'),
    ('rf', '随机森林'),
    ('lr', '逻辑回归'),
    ('textcnn', 'TextCNN'),
    ('textrcnn', 'TextRCNN'),
)


def _metrics_json_dict(mi):
    if not mi or not (mi.metrics_json or '').strip():
        return {}
    try:
        d = json.loads(mi.metrics_json)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _pick_float(*candidates):
    for c in candidates:
        if c is None:
            continue
        try:
            if isinstance(c, str) and not c.strip():
                continue
            return float(c)
        except (TypeError, ValueError):
            continue
    return None


def _overfit_from_json(d):
    if not d:
        return None
    if 'is_overfitting' in d:
        v = d.get('is_overfitting')
        if v is True:
            return True
        if v is False:
            return False
        sv = str(v).strip().lower()
        if sv in ('1', 'true', 'yes', '是'):
            return True
        if sv in ('0', 'false', 'no', '否'):
            return False
    s = d.get('overfitting')
    if isinstance(s, str):
        s = s.strip()
        if s in ('是', 'yes', 'true', '1', '过拟合'):
            return True
        if s in ('否', 'no', 'false', '0', '未过拟合'):
            return False
    return None


def _format_train_sec(sec):
    if sec is None:
        return None
    try:
        s = float(sec)
    except (TypeError, ValueError):
        return None
    if s < 0:
        return None
    if s < 60:
        return '{:.1f} 秒'.format(s)
    if s < 3600:
        return '{:.1f} 分钟'.format(s / 60.0)
    return '{:.2f} 小时'.format(s / 3600.0)


def _fmt_metric_4(x):
    if x is None:
        return '—'
    return '{:.4f}'.format(float(x))


def _build_compare_row_cells(ev, mi):
    """
    从最新评估记录 ev 与兜底 ModelInfo mi 组装对比表指标单元（与评估记录、metrics_json 优先级一致）。
    ev 可为 None；mi 可为 None（则仅 mj_mi 为空 dict）。
    """
    mj_ev = _metrics_json_dict(ev.model_info) if ev else {}
    mj_mi = _metrics_json_dict(mi) if mi else {}

    acc = _pick_float(
        ev.accuracy if ev else None,
        mj_ev.get('accuracy'),
        mj_mi.get('accuracy'),
    )
    prec = _pick_float(
        ev.precision if ev else None,
        mj_ev.get('precision'),
        mj_mi.get('precision'),
    )
    rec = _pick_float(
        ev.recall if ev else None,
        mj_ev.get('recall'),
        mj_mi.get('recall'),
    )
    f1v = _pick_float(
        ev.f1_score if ev else None,
        mj_ev.get('f1_score'),
        mj_ev.get('f1'),
        mj_mi.get('f1_score'),
        mj_mi.get('f1'),
    )
    auc = _pick_float(
        ev.auc if ev else None,
        mj_ev.get('auc'),
        mj_ev.get('roc_auc'),
        mj_mi.get('auc'),
        mj_mi.get('roc_auc'),
    )
    train_sec = _pick_float(
        ev.training_time_sec if ev else None,
        mj_ev.get('training_time_sec'),
        mj_ev.get('train_seconds'),
        mj_mi.get('training_time_sec'),
        mj_mi.get('train_seconds'),
    )

    over = None
    if ev and ev.is_overfitting is not None:
        over = ev.is_overfitting
    if over is None:
        over = _overfit_from_json(mj_ev)
    if over is None:
        over = _overfit_from_json(mj_mi)

    over_txt = '—'
    if over is True:
        over_txt = '是'
    elif over is False:
        over_txt = '否'

    return {
        'accuracy': _fmt_metric_4(acc),
        'precision': _fmt_metric_4(prec),
        'recall': _fmt_metric_4(rec),
        'f1': _fmt_metric_4(f1v),
        'auc': _fmt_metric_4(auc),
        'train_time': _format_train_sec(train_sec) or '—',
        'overfitting': over_txt,
        'eval_at': ev.evaluated_at if ev else None,
        'model_label': (
            '{} / {}'.format(ev.model_info.name, ev.model_version)
            if ev
            else (mi.name if mi else None)
        ),
    }


def build_algorithm_compare_rows():
    """源算法：按七种 model_type 各取最新一条 ModelEvaluation，缺项用对应 ModelInfo.metrics_json 补齐。"""
    rows = []
    for key, display_name in ALGO_COMPARE_DEFS:
        ev = (
            ModelEvaluation.objects.filter(model_info__model_type=key)
            .select_related('model_info')
            .order_by('-evaluated_at')
            .first()
        )
        mi = ModelInfo.objects.filter(model_type=key).order_by('-is_active', '-created_at').first()
        cells = _build_compare_row_cells(ev, mi)
        rows.append(
            {
                'key': key,
                'name': display_name,
                **cells,
            }
        )
    return rows


def build_fusion_strategy_compare_rows():
    """
    融合策略宽表：默认验证集上按基模型类概率加权融合，输出分类与概率校准类指标（见 fusion_strategy_report）。
    排除占位「线下评估记录 / default」。
    """
    rows = []
    qs = (
        ModelInfo.objects.filter(model_type='fusion')
        .exclude(name=EVAL_RECORD_NAME, version=EVAL_RECORD_VERSION)
        .order_by('-is_active', '-created_at', '-id')
    )
    for mi in qs:
        rep = get_cached_fusion_strategy_report(mi)
        row = report_to_table_row(rep)
        row['id'] = mi.id
        rows.append(row)
    return rows


def _fusion_bool_keys():
    return {k for k, v in PredictService.DEFAULT_RUNTIME_CONFIG.items() if isinstance(v, bool)}


def _cfg_checkbox_checked(raw):
    return str(raw).lower() in ('1', 'true', 'yes', 'on')


_FUSION_KEY_TITLE = {
    'enable_svm': 'SVM（支持向量机）',
    'enable_knn': 'K 近邻（KNN）',
    'enable_rf': '随机森林',
    'enable_dt': '决策树',
    'enable_lr': '逻辑回归',
    'enable_textcnn': 'TextCNN（深度学习）',
    'enable_textrcnn': 'TextRCNN（深度学习）',
    'enable_rules': '关键词规则',
    'weight_svm': 'SVM',
    'weight_knn': 'KNN',
    'weight_rf': '随机森林',
    'weight_dt': '决策树',
    'weight_lr': '逻辑回归',
    'weight_textcnn': 'TextCNN',
    'weight_textrcnn': 'TextRCNN',
    'weight_rules': '规则',
    'threshold_high_risk': '高风险分数线',
    'threshold_alert': '需要留意的分数线',
    'threshold_rule_alert': '只靠规则时的提醒线',
    'tfidf_vectorizer_path': '词袋/TF-IDF 向量文件',
    'svm_model_path': 'SVM 模型文件',
    'knn_model_path': 'KNN 模型文件',
    'rf_model_path': '随机森林模型文件',
    'dt_model_path': '决策树模型文件',
    'lr_model_path': '逻辑回归模型文件',
    'textcnn_vocab_path': 'TextCNN 词表',
    'textcnn_weight_path': 'TextCNN 权重',
    'textrcnn_vocab_path': 'TextRCNN 词表',
    'textrcnn_weight_path': 'TextRCNN 权重',
    'num2name_path': '类别编号对照表',
    'textcnn_max_len': '每条文本最长字符数',
    'textcnn_embedding_dim': '向量维度',
    'textcnn_num_classes': '分类类别数',
    'textcnn_kernel_sizes': '卷积核大小（逗号分隔）',
    'textcnn_num_channels': '卷积通道数',
    'textcnn_dropout': 'Dropout',
    'textrcnn_max_len': '每条文本最长字符数',
    'textrcnn_embedding_dim': '向量维度',
    'textrcnn_hidden_dim': '隐藏层大小',
    'textrcnn_num_classes': '分类类别数',
    'textrcnn_dropout': 'Dropout',
}

_FUSION_KEY_HELP = {
    'enable_svm': '常用、速度快，适合先做基线。',
    'enable_rules': '按敏感词等规则加分，可与模型结果一起综合。',
    'weight_svm': 'SVM 在最终综合分里占多大比重。',
    'weight_rules': '规则分在综合分里占多大比重。',
    'threshold_high_risk': '综合分 ≥ 此值时，可视为高风险（可按业务调严或放宽）。',
    'threshold_alert': '综合分 ≥ 此值时，可提示需要关注。',
    'threshold_rule_alert': '仅规则参与时，规则分 ≥ 此值才提醒。',
    'tfidf_vectorizer_path': '一般为 model/svm/tfidfVectorizer.pkl（与各算法子目录下向量器一致），需与当时训练一致。',
    'num2name_path': '一般为 model/num2name.json。',
}

AUTO_PRESET_NAME = '【自动】上次保存的融合配置'
_MAX_NAMED_PRESETS = 40


def _default_value_type_for_key(k):
    d = PredictService.DEFAULT_RUNTIME_CONFIG.get(k)
    return type(d).__name__ if d is not None else 'str'


def _build_fusion_snapshot_from_db():
    PredictService.ensure_default_runtime_config()
    cfg = {r.key: r.value for r in SystemConfig.objects.all().order_by('key')}
    active = ModelInfo.objects.filter(is_active=True, status='ready').order_by('-created_at').first()
    out = {
        'version': 1,
        'system_config': cfg,
        'active_model_id': active.id if active else None,
    }
    if active:
        out['active_model_label'] = '{} / {}'.format(active.name, active.version)
        out['active_model_type'] = active.model_type
    return out


def _norm_model_rel_path(p):
    s = (p or '').strip().replace('\\', '/')
    return s[:255] if s else ''


_FUSION_ROLE_LABELS = {
    'svm': 'SVM',
    'knn': 'KNN',
    'rf': '随机森林',
    'dt': '决策树',
    'lr': '逻辑回归',
    'textcnn': 'TextCNN',
    'textrcnn': 'TextRCNN',
    'rules': '规则',
}
_FUSION_ROLE_ORDER = ('svm', 'knn', 'rf', 'dt', 'lr', 'textcnn', 'textrcnn', 'rules')


def _fusion_strategy_list_summary(mi):
    """模型列表「文件路径」列：展示融合策略登记的权重比率与启用子模型。"""
    if not is_canonical_fusion_registry_row(mi):
        return None
    try:
        root = json.loads(mi.metrics_json or '{}')
    except Exception:
        return None
    snap = root.get('fusion_snapshot')
    if not isinstance(snap, dict):
        return None
    ratios = snap.get('weight_ratios_normalized') or {}
    ratio_parts = []
    for role in _FUSION_ROLE_ORDER:
        r = ratios.get(role)
        if r is None:
            continue
        try:
            ratio_parts.append(
                '{} {:.1%}'.format(_FUSION_ROLE_LABELS.get(role, role), float(r)),
            )
        except (TypeError, ValueError):
            continue
    line1 = (
        '融合权重比率（启用项归一）: ' + '，'.join(ratio_parts)
        if ratio_parts
        else '融合权重比率: —'
    )
    enables = snap.get('enables') or {}
    on = [
        _FUSION_ROLE_LABELS[r]
        for r in _FUSION_ROLE_ORDER
        if enables.get('enable_{}'.format(r))
    ]
    line2 = '当前启用: ' + '、'.join(on) if on else '当前启用: —'
    return line1 + '\n' + line2


def _build_fusion_snapshot_dict_for_cfg(cfg):
    """由已合并的运行时 cfg 生成 metrics_json 内 fusion_snapshot 的数据体。"""
    PredictService.ensure_default_runtime_config()

    def _vec_sklearn(mt):
        row = SystemConfig.objects.filter(key='tfidf_vectorizer_path_{}'.format(mt)).first()
        if row and (row.value or '').strip():
            return _norm_model_rel_path(row.value)
        return _norm_model_rel_path(cfg.get('tfidf_vectorizer_path'))

    spec = [
        ('svm', cfg.get('svm_model_path'), cfg.get('tfidf_vectorizer_path')),
        ('knn', cfg.get('knn_model_path'), None),
        ('rf', cfg.get('rf_model_path'), None),
        ('dt', cfg.get('dt_model_path'), None),
        ('lr', cfg.get('lr_model_path'), None),
        ('textcnn', cfg.get('textcnn_weight_path'), cfg.get('textcnn_vocab_path')),
        ('textrcnn', cfg.get('textrcnn_weight_path'), cfg.get('textrcnn_vocab_path')),
    ]

    paths = {}
    for mt, fp_src, vp_src in spec:
        fp = _norm_model_rel_path(fp_src)
        vp = _norm_model_rel_path(vp_src) if vp_src is not None else _vec_sklearn(mt)
        paths[mt] = {'file_path': fp or None, 'vectorizer_path': vp or None}

    keys_weight = [k for k in sorted(PredictService.DEFAULT_RUNTIME_CONFIG) if k.startswith('weight_')]
    keys_enable = [k for k in sorted(PredictService.DEFAULT_RUNTIME_CONFIG) if k.startswith('enable_')]
    weights = {k: cfg.get(k) for k in keys_weight}
    enables = {k: cfg.get(k) for k in keys_enable}

    role_meta = [
        ('svm', 'weight_svm', 'enable_svm'),
        ('knn', 'weight_knn', 'enable_knn'),
        ('rf', 'weight_rf', 'enable_rf'),
        ('dt', 'weight_dt', 'enable_dt'),
        ('lr', 'weight_lr', 'enable_lr'),
        ('textcnn', 'weight_textcnn', 'enable_textcnn'),
        ('textrcnn', 'weight_textrcnn', 'enable_textrcnn'),
        ('rules', 'weight_rules', 'enable_rules'),
    ]
    total_w = 0.0
    for _role, wk, ek in role_meta:
        try:
            wv = float(cfg.get(wk, 0) or 0)
        except (TypeError, ValueError):
            wv = 0.0
        if cfg.get(ek) and wv > 0:
            total_w += wv
    ratios_norm = {}
    for role, wk, ek in role_meta:
        try:
            wv = float(cfg.get(wk, 0) or 0)
        except (TypeError, ValueError):
            wv = 0.0
        if cfg.get(ek) and wv > 0 and total_w > 0:
            ratios_norm[role] = wv / total_w

    return {
        'paths': paths,
        'weights': weights,
        'enables': enables,
        'weight_ratios_normalized': ratios_norm,
        'updated_at': timezone.now().isoformat(),
    }


def _create_autonamed_fusion_registration_after_runtime_save():
    """
    「预测融合设置」保存成功后：按 策略模型01/02… 自动命名新增一条融合登记，
    metrics_json 写入当前全局 runtime 对应的 fusion_snapshot（与主登记「融合策略」内容一致）。
    """
    from app.registry_names import next_autonamed_fusion_strategy_model_name

    PredictService.ensure_default_runtime_config()
    cfg = PredictService.get_runtime_config(None)
    snap_d = _build_fusion_snapshot_dict_for_cfg(cfg)
    name = next_autonamed_fusion_strategy_model_name()
    ModelInfo.objects.create(
        name=name,
        version=FUSION_STRATEGY_MODELINFO_VERSION,
        model_type=FUSION_STRATEGY_MODEL_TYPE,
        status='ready',
        is_active=False,
        listed_for_users=False,
        file_path=None,
        vectorizer_path=None,
        description='由「预测融合设置」保存自动生成',
        config_json='{}',
        metrics_json=json.dumps({'fusion_snapshot': snap_d}, ensure_ascii=False, indent=2),
    )
    return name


def sync_modelinfo_paths_from_fusion_config():
    """
    维护名称严格为「融合策略」、version=default 的登记：在 metrics_json 汇总
    各子模型路径与 enable/weight。其他名称的融合策略登记不会被本函数覆盖。在线预测对任一条融合登记会合并其 metrics_json.fusion_snapshot。
    """
    PredictService.ensure_default_runtime_config()
    cfg = PredictService.get_runtime_config(None)
    snapshot = _build_fusion_snapshot_dict_for_cfg(cfg)
    snapshot_json = json.dumps({'fusion_snapshot': snapshot}, ensure_ascii=False, indent=2)
    desc = (
        '全局快照登记（名称须为「融合策略」）：汇总「预测融合设置」；启用项归一化比率见 fusion_snapshot。'
    )

    qs = ModelInfo.objects.filter(
        model_type=FUSION_STRATEGY_MODEL_TYPE,
        version=FUSION_STRATEGY_MODELINFO_VERSION,
        name=FUSION_STRATEGY_MODELINFO_NAME,
    )
    if not qs.exists():
        ModelInfo.objects.create(
            name=FUSION_STRATEGY_MODELINFO_NAME,
            version=FUSION_STRATEGY_MODELINFO_VERSION,
            model_type=FUSION_STRATEGY_MODEL_TYPE,
            status='ready',
            is_active=False,
            listed_for_users=True,
            file_path=None,
            vectorizer_path=None,
            description=desc,
            config_json='{}',
            metrics_json=snapshot_json,
        )
    else:
        qs.update(
            model_type=FUSION_STRATEGY_MODEL_TYPE,
            file_path=None,
            vectorizer_path=None,
            metrics_json=snapshot_json,
            description=desc,
        )


def _apply_fusion_snapshot(payload, apply_active_model=True):
    sc = payload.get('system_config') if isinstance(payload, dict) else None
    if not isinstance(sc, dict):
        raise ValueError('快照格式无效：缺少 system_config')
    for k, v in sc.items():
        if not k:
            continue
        ks = str(k)[:64]
        vs = str(v) if v is not None else ''
        vs = vs[:255]
        row = SystemConfig.objects.filter(key=ks).first()
        if row:
            row.value = vs
            row.save(update_fields=['value', 'updated_at'])
        else:
            SystemConfig.objects.create(
                key=ks,
                value=vs,
                value_type=_default_value_type_for_key(ks),
                description='预测融合运行配置',
            )
    if apply_active_model:
        mid = payload.get('active_model_id')
        if mid is not None:
            try:
                mid = int(mid)
            except (TypeError, ValueError):
                mid = None
        if mid:
            m = ModelInfo.objects.filter(id=mid, status='ready').first()
            if m:
                ModelInfo.objects.all().update(is_active=False)
                m.is_active = True
                m.save(update_fields=['is_active'])

    sync_modelinfo_paths_from_fusion_config()


def _touch_autosave_preset():
    body = json.dumps(_build_fusion_snapshot_from_db(), ensure_ascii=False)
    autos = list(FusionConfigPreset.objects.filter(is_auto=True).order_by('id'))
    for row in autos[1:]:
        row.delete()
    if autos:
        first = FusionConfigPreset.objects.filter(id=autos[0].id).first()
        if first:
            first.name = AUTO_PRESET_NAME
            first.remark = '在「预测融合设置」中每次点击「保存设置」时自动更新'
            first.snapshot_json = body
            first.save(update_fields=['name', 'remark', 'snapshot_json'])
    else:
        FusionConfigPreset.objects.create(
            is_auto=True,
            name=AUTO_PRESET_NAME,
            remark='在「预测融合设置」中每次点击「保存设置」时自动更新',
            snapshot_json=body,
        )


def _trim_named_presets():
    qs = FusionConfigPreset.objects.filter(is_auto=False).order_by('-created_at')
    ids = list(qs.values_list('id', flat=True))
    if len(ids) <= _MAX_NAMED_PRESETS:
        return
    for pk in ids[_MAX_NAMED_PRESETS:]:
        FusionConfigPreset.objects.filter(id=pk).delete()


def _fusion_presets_for_template():
    return {
        'auto_preset': FusionConfigPreset.objects.filter(is_auto=True).first(),
        'named_presets': list(FusionConfigPreset.objects.filter(is_auto=False).order_by('-updated_at')[:30]),
        'auto_preset_name': AUTO_PRESET_NAME,
    }


def _runtime_summary_sections(cfg):
    if not cfg:
        return []
    sections = []
    enabled = []
    for k in sorted(_fusion_bool_keys()):
        if cfg.get(k):
            enabled.append(_FUSION_KEY_TITLE.get(k, k))
    sections.append({'title': '参与融合的子模型', 'kind': 'tags', 'items': enabled})
    weights = []
    for k in [
        'weight_svm',
        'weight_knn',
        'weight_rf',
        'weight_dt',
        'weight_lr',
        'weight_textcnn',
        'weight_textrcnn',
        'weight_rules',
    ]:
        try:
            w = float(cfg.get(k, 0) or 0)
        except (TypeError, ValueError):
            w = 0.0
        if w > 0:
            weights.append({'label': _FUSION_KEY_TITLE.get(k, k), 'value': '{:.4f}'.format(w)})
    sections.append({'title': '非零权重', 'kind': 'table', 'items': weights})
    th = []
    for k in ['threshold_high_risk', 'threshold_alert', 'threshold_rule_alert']:
        th.append({'label': _FUSION_KEY_TITLE.get(k, k), 'value': cfg.get(k)})
    sections.append({'title': '风险门槛', 'kind': 'table', 'items': th})
    return sections


def _build_fusion_config_sections(rows):
    """把 SystemConfig 行分成易读区块，供融合配置页使用。"""
    by_key = {r.key: r for r in rows}
    bool_keys = _fusion_bool_keys()

    def _path_prefix_for_key(k):
        """为前端文件选择器提供相对路径前缀。"""
        if k == 'tfidf_vectorizer_path':
            return 'model/svm/'
        if k in ('svm_model_path', 'tfidf_vectorizer_path_svm'):
            return 'model/svm/'
        if k == 'knn_model_path':
            return 'model/knn/'
        if k == 'rf_model_path':
            return 'model/rf/'
        if k == 'dt_model_path':
            return 'model/dt/'
        if k == 'lr_model_path':
            return 'model/lr/'
        if k in ('textcnn_weight_path', 'textcnn_vocab_path'):
            return 'model/textcnn/'
        if k in ('textrcnn_weight_path', 'textrcnn_vocab_path'):
            return 'model/textrcnn/'
        if k == 'num2name_path':
            return 'model/'
        return 'model/'

    section_defs = [
        (
            'switch',
            '① 要用哪些模型一起判断',
            '勾选后才会参与「综合预测」。新手可只开 SVM + 规则；训练过深度学习再打开对应项。',
            [
                'enable_svm',
                'enable_knn',
                'enable_rf',
                'enable_dt',
                'enable_lr',
                'enable_textcnn',
                'enable_textrcnn',
                'enable_rules',
            ],
        ),
        (
            'weight',
            '② 各模型话语权（权重）',
            '0～1 之间，越大越「听它的」。不必加起来等于 1。',
            [
                'weight_svm',
                'weight_knn',
                'weight_rf',
                'weight_dt',
                'weight_lr',
                'weight_textcnn',
                'weight_textrcnn',
                'weight_rules',
            ],
        ),
        (
            'threshold',
            '③ 风险与提醒门槛',
            '综合得分在 0～1 之间，超过下面分数线会产生不同级别的提示。',
            ['threshold_high_risk', 'threshold_alert', 'threshold_rule_alert'],
        ),
        (
            'paths',
            '高级 · 模型文件路径（融合策略模型）',
            '与「模型管理」中唯一一条「融合策略」登记（类型为融合策略模型）内的路径快照一致；保存本页会更新该条 metrics_json。未改训练产物则不用动。',
            [
                'tfidf_vectorizer_path',
                'svm_model_path',
                'knn_model_path',
                'rf_model_path',
                'dt_model_path',
                'lr_model_path',
                'textcnn_vocab_path',
                'textcnn_weight_path',
                'textrcnn_vocab_path',
                'textrcnn_weight_path',
                'num2name_path',
            ],
        ),
        (
            'deep',
            '高级 · 深度学习结构参数',
            '须与训练该模型时一致；日常只调开关和权重即可，不必改这里。',
            [
                'textcnn_max_len',
                'textcnn_embedding_dim',
                'textcnn_num_classes',
                'textcnn_kernel_sizes',
                'textcnn_num_channels',
                'textcnn_dropout',
                'textrcnn_max_len',
                'textrcnn_embedding_dim',
                'textrcnn_hidden_dim',
                'textrcnn_num_classes',
                'textrcnn_dropout',
            ],
        ),
    ]

    sections_out = []
    used = set()

    for sid, title, subtitle, keys in section_defs:
        items = []
        for k in keys:
            if k not in by_key:
                continue
            used.add(k)
            row = by_key[k]
            default = PredictService.DEFAULT_RUNTIME_CONFIG.get(k)
            if k in bool_keys:
                widget = 'checkbox'
                step = None
            elif k.startswith('weight_') or k.startswith('threshold_'):
                widget = 'number'
                step = '0.05'
            elif isinstance(default, float):
                widget = 'number'
                step = '0.05'
            elif isinstance(default, int):
                widget = 'number'
                step = '1'
            else:
                widget = 'text'
                step = None
            items.append(
                {
                    'key': k,
                    'row': row,
                    'title': _FUSION_KEY_TITLE.get(k, k),
                    'help': _FUSION_KEY_HELP.get(k, ''),
                    'widget': widget,
                    'step': step,
                    'path_prefix': _path_prefix_for_key(k),
                    'checked': _cfg_checkbox_checked(row.value),
                }
            )
        sections_out.append({'id': sid, 'title': title, 'subtitle': subtitle, 'items': items})

    rest_keys = sorted(set(by_key) - used)
    if rest_keys:
        rest_items = []
        for k in rest_keys:
            row = by_key[k]
            default = PredictService.DEFAULT_RUNTIME_CONFIG.get(k)
            if k in bool_keys:
                w, step, checked = 'checkbox', None, _cfg_checkbox_checked(row.value)
            elif isinstance(default, float):
                w, step, checked = 'number', '0.05', False
            elif isinstance(default, int):
                w, step, checked = 'number', '1', False
            else:
                w, step, checked = 'text', None, False
            rest_items.append(
                {
                    'key': k,
                    'row': row,
                    'title': _FUSION_KEY_TITLE.get(k, k),
                    'help': '',
                    'widget': w,
                    'step': step,
                    'path_prefix': _path_prefix_for_key(k),
                    'checked': checked,
                }
            )
        sections_out.append(
            {
                'id': 'other',
                'title': '其他',
                'subtitle': '扩展项，一般无需修改。',
                'items': rest_items,
            }
        )

    return sections_out


def _get_login_userinfo(request):
    return get_admin_panel_user(request)


def _list_model_directory_files():
    """递归列出项目 model/ 下文件（含子目录），并合并 model_training_registry.json 中的训练指标。"""
    from algorithm.model_training_registry import REGISTRY_FILENAME, load_registry

    root = os.path.join(settings.BASE_DIR, 'model')
    if not os.path.isdir(root):
        return []
    reg = load_registry(settings.BASE_DIR)
    out = []
    root_norm = os.path.normpath(root)
    allowed_ext = {'.pkl', '.joblib', '.pt', '.pth', '.json', '.txt', '.csv', '.bin'}
    known_top_dirs = {'svm', 'knn', 'rf', 'dt', 'lr', 'textcnn', 'textrcnn', 'legacy'}
    keep_root_files = {'model/num2name.json'}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name == REGISTRY_FILENAME and os.path.normpath(dirpath) == root_norm:
                continue
            path = os.path.join(dirpath, name)
            if not os.path.isfile(path):
                continue
            st = os.stat(path)
            rel = os.path.relpath(path, settings.BASE_DIR).replace(os.sep, '/')
            # 只展示“模型相关产物”，避免把 training 中间文件也刷到列表里
            ext = os.path.splitext(name)[1].lower()
            if ext and ext not in allowed_ext:
                continue
            if rel not in keep_root_files:
                parts = (rel or '').split('/')
                if len(parts) < 3:
                    continue
                if parts[1] not in known_top_dirs:
                    continue
            meta = reg.get(rel) or {}
            br = meta.get('best_recall_macro')
            ll = meta.get('last_loss')
            item = {
                'name': name,
                'rel': rel,
                'size': st.st_size,
                'mtime': st.st_mtime,
                'mtime_display': datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                'train_count': int(meta.get('train_count') or 0),
                'best_recall_display': None,
                'last_loss_display': None,
                'last_train_at': meta.get('last_train_at') or '',
                'last_algo': meta.get('last_algo') or '',
                'last_csv': meta.get('last_csv') or '',
            }
            if br is not None:
                try:
                    item['best_recall_display'] = '{:.4f}'.format(float(br))
                except (TypeError, ValueError):
                    pass
            if ll is not None:
                try:
                    item['last_loss_display'] = '{:.4f}'.format(float(ll))
                except (TypeError, ValueError):
                    pass
            out.append(item)
    out.sort(key=lambda x: (-x['mtime'], x['rel'].lower()))
    return out


@method_decorator(check_admin_access, name='dispatch')
class ModelListView(View):
    def get(self, request):
        userinfo = _get_login_userinfo(request)
        rows = ModelInfo.objects.all().order_by('-created_at')
        active = rows.filter(is_active=True, status='ready').first()
        rows_display = [
            {
                'row': r,
                'registry_label': registry_display_label(r, EVAL_RECORD_NAME, EVAL_RECORD_VERSION),
                'fusion_strategy_summary': _fusion_strategy_list_summary(r),
            }
            for r in rows
        ]
        disk_files = _list_model_directory_files()
        return render(
            request,
            'model/model_list.html',
            {
                'userinfo': userinfo,
                'rows_display': rows_display,
                'active': active,
                'model_disk_files': disk_files,
                'model_dir_rel': 'model',
            },
        )

    def post(self, request):
        action = request.POST.get('action')
        model_id = request.POST.get('model_id')
        row = get_object_or_404(ModelInfo, id=model_id)
        # 操作完成后带锚点返回，避免整页回到顶部、连续操作还要再滚下去
        scroll_anchor = '#model-reg-{}'.format(row.id)
        if action == 'enable':
            ModelInfo.objects.all().update(is_active=False)
            row.is_active = True
            row.status = 'ready'
            row.save(update_fields=['is_active', 'status'])
            messages.info(request, '模型启用成功（仅切换默认模型，不自动开放前台）')
        elif action == 'disable':
            row.is_active = False
            row.save(update_fields=['is_active'])
            messages.info(request, '模型已停用')
        elif action == 'listed_show':
            row.listed_for_users = True
            row.save(update_fields=['listed_for_users'])
            messages.info(request, '已开放：前台预测与用户管理中可选择该类型')
        elif action == 'listed_hide':
            row.listed_for_users = False
            row.save(update_fields=['listed_for_users'])
            messages.info(request, '已关闭前台展示（登记仍保留，后台仍可见）')
        elif action == 'delete':
            label = f'{row.name} / {row.version}'
            row.delete()
            scroll_anchor = '#db-model-registry'
            messages.info(request, f'已删除模型登记：{label}（关联的模型评估记录会一并删除；预测结果等仍保留，模型外键将置空）')
        else:
            messages.warning(request, '未知操作')
        return redirect(reverse('model_list') + scroll_anchor)


def _fusion_path_related_keys():
    s = set()
    for k in PredictService.DEFAULT_RUNTIME_CONFIG:
        if '_path' in k or k.startswith('tfidf_vectorizer_path'):
            s.add(k)
    for role in ('svm', 'knn', 'rf', 'dt', 'lr'):
        s.add('tfidf_vectorizer_path_{}'.format(role))
    return frozenset(s)


def _is_known_runtime_config_key(k):
    if k in PredictService.DEFAULT_RUNTIME_CONFIG:
        return True
    if isinstance(k, str) and k.startswith('tfidf_vectorizer_path_'):
        return True
    return False


def _coerce_value_for_system_config(key, raw):
    default = PredictService.DEFAULT_RUNTIME_CONFIG.get(key)
    if default is not None:
        if isinstance(default, bool):
            return 'True' if str(raw).lower() in ('1', 'true', 'yes', 'on') else 'False'
        if isinstance(default, int):
            try:
                return str(int(raw))
            except Exception:
                return str(int(default))
        if isinstance(default, float):
            try:
                return str(float(raw))
            except Exception:
                return str(float(default))
    if isinstance(raw, bool):
        return 'True' if raw else 'False'
    s = str(raw).strip() if raw is not None else ''
    return s[:255]


def _system_config_upsert(key, value_str):
    ks = str(key)[:64]
    vs = value_str if value_str is not None else ''
    if len(vs) > 255:
        vs = vs[:255]
    row = SystemConfig.objects.filter(key=ks).first()
    if row:
        row.value = vs
        row.save(update_fields=['value', 'updated_at'])
    else:
        SystemConfig.objects.create(
            key=ks,
            value=vs,
            value_type=_default_value_type_for_key(ks),
            description='预测融合运行配置',
        )


def _extended_runtime_config_for_edit():
    """与预测融合相关的 SystemConfig 全量读出（含 DEFAULT 未列出的扩展键），供编辑页与 JSON 对齐。"""
    PredictService.ensure_default_runtime_config()
    base = PredictService.get_runtime_config(None)
    out = dict(base)
    for row in SystemConfig.objects.all().order_by('key'):
        if row.key not in out:
            out[row.key] = row.value
    return out


def _paths_from_runtime_for_model_type(mt, cfg):
    if mt not in SOURCE_ALGO_TYPES:
        return '', ''

    def _vec_sklearn(m):
        row = SystemConfig.objects.filter(key='tfidf_vectorizer_path_{}'.format(m)).first()
        if row and (row.value or '').strip():
            return _norm_model_rel_path(row.value)
        return _norm_model_rel_path(cfg.get('tfidf_vectorizer_path'))

    spec = {
        'svm': (cfg.get('svm_model_path'), cfg.get('tfidf_vectorizer_path')),
        'knn': (cfg.get('knn_model_path'), None),
        'rf': (cfg.get('rf_model_path'), None),
        'dt': (cfg.get('dt_model_path'), None),
        'lr': (cfg.get('lr_model_path'), None),
        'textcnn': (cfg.get('textcnn_weight_path'), cfg.get('textcnn_vocab_path')),
        'textrcnn': (cfg.get('textrcnn_weight_path'), cfg.get('textrcnn_vocab_path')),
    }
    fp_src, vp_src = spec[mt]
    fp = _norm_model_rel_path(fp_src)
    vp = _norm_model_rel_path(vp_src) if vp_src is not None else _vec_sklearn(mt)
    return fp or '', vp or ''


def _write_algo_paths_to_system_config(mt, fp, vp):
    if mt not in SOURCE_ALGO_TYPES:
        return
    fp = _norm_model_rel_path(fp)
    vp = _norm_model_rel_path(vp)
    if mt == 'svm':
        _system_config_upsert('svm_model_path', fp)
        _system_config_upsert('tfidf_vectorizer_path', vp)
        _system_config_upsert('tfidf_vectorizer_path_svm', vp)
    elif mt == 'knn':
        _system_config_upsert('knn_model_path', fp)
        _system_config_upsert('tfidf_vectorizer_path_knn', vp)
    elif mt == 'rf':
        _system_config_upsert('rf_model_path', fp)
        _system_config_upsert('tfidf_vectorizer_path_rf', vp)
    elif mt == 'dt':
        _system_config_upsert('dt_model_path', fp)
        _system_config_upsert('tfidf_vectorizer_path_dt', vp)
    elif mt == 'lr':
        _system_config_upsert('lr_model_path', fp)
        _system_config_upsert('tfidf_vectorizer_path_lr', vp)
    elif mt == 'textcnn':
        _system_config_upsert('textcnn_weight_path', fp)
        _system_config_upsert('textcnn_vocab_path', vp)
    elif mt == 'textrcnn':
        _system_config_upsert('textrcnn_weight_path', fp)
        _system_config_upsert('textrcnn_vocab_path', vp)


def _apply_model_edit_config_json_post(raw_json, skip_path_keys, write_all_keys):
    """
    将模型编辑页的「模型配置JSON」写回 SystemConfig；无法识别的键写入返回值以便存 ModelInfo.config_json。
    skip_path_keys=True：源算法登记时路径以独立输入框为准，不从 JSON 改路径键。
    write_all_keys=True：融合策略登记时整份 JSON 与预测融合设置对齐写入。
    """
    extras = {}
    raw_json = (raw_json or '').strip()
    try:
        parsed = json.loads(raw_json) if raw_json else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        raise ValueError('模型配置JSON 不是合法 JSON')
    if not isinstance(parsed, dict):
        raise ValueError('模型配置JSON 须为 JSON 对象')
    path_keys = _fusion_path_related_keys()
    for k, v in parsed.items():
        if not k:
            continue
        k = str(k).strip()[:64]
        if not k:
            continue
        if write_all_keys:
            if _is_known_runtime_config_key(k) or k in path_keys:
                _system_config_upsert(k, _coerce_value_for_system_config(k, v))
            else:
                extras[k] = v
        else:
            if skip_path_keys and k in path_keys:
                continue
            if _is_known_runtime_config_key(k):
                _system_config_upsert(k, _coerce_value_for_system_config(k, v))
            else:
                extras[k] = v
    return extras


def _model_edit_model_type_locked(row):
    """规范登记不允许改 model_type，避免与训练/融合逻辑错位；名称可手动修改。"""
    if is_canonical_fusion_registry_row(row):
        return True
    if (row.name or '').strip() in source_algo_modelinfo_name_set() and row.model_type in SOURCE_ALGO_TYPES:
        return True
    return False


@method_decorator(check_admin_access, name='dispatch')
class ModelEditView(View):
    def get(self, request, model_id=None):
        if model_id is None:
            messages.warning(request, '当前已关闭「新增模型」入口，请通过「模型训练」生成文件，并在「融合配置」中指定路径。')
            return redirect('model_list')
        userinfo = _get_login_userinfo(request)
        row = ModelInfo.objects.filter(id=model_id).first()
        if not row:
            messages.warning(request, '未找到该模型记录')
            return redirect('model_list')

        PredictService.ensure_default_runtime_config()
        cfg = _extended_runtime_config_for_edit()
        path_keys = _fusion_path_related_keys()
        is_fusion_strategy = is_canonical_fusion_registry_row(row)

        if is_fusion_strategy:
            form_file_path = ''
            form_vectorizer_path = ''
            merged = {}
            for k in sorted(cfg.keys()):
                if isinstance(k, str):
                    merged[k] = cfg[k]
            try:
                overlay = json.loads(row.config_json or '{}')
                if isinstance(overlay, dict):
                    for k, v in overlay.items():
                        merged[k] = v
            except Exception:
                pass
            form_config_json = json.dumps(merged, ensure_ascii=False, indent=2)
        elif row.model_type in SOURCE_ALGO_TYPES:
            rt = PredictService.get_runtime_config(None)
            form_file_path, form_vectorizer_path = _paths_from_runtime_for_model_type(row.model_type, rt)
            merged = {}
            for k in sorted(PredictService.DEFAULT_RUNTIME_CONFIG.keys()):
                if k not in path_keys:
                    merged[k] = cfg[k]
            for k in sorted(cfg.keys()):
                if k in path_keys or k in merged:
                    continue
                if _is_known_runtime_config_key(k):
                    merged[k] = cfg[k]
            try:
                overlay = json.loads(row.config_json or '{}')
                if isinstance(overlay, dict):
                    for k, v in overlay.items():
                        if k not in path_keys:
                            merged[k] = v
            except Exception:
                pass
            form_config_json = json.dumps(merged, ensure_ascii=False, indent=2)
        else:
            form_file_path = row.file_path or ''
            form_vectorizer_path = row.vectorizer_path or ''
            merged = {}
            for k in sorted(PredictService.DEFAULT_RUNTIME_CONFIG.keys()):
                if k not in path_keys:
                    merged[k] = cfg[k]
            try:
                overlay = json.loads(row.config_json or '{}')
                if isinstance(overlay, dict):
                    for k, v in overlay.items():
                        if k not in path_keys:
                            merged[k] = v
            except Exception:
                pass
            form_config_json = json.dumps(merged, ensure_ascii=False, indent=2)

        model_type_locked = _model_edit_model_type_locked(row)

        return render(
            request,
            'model/model_edit.html',
            {
                'userinfo': userinfo,
                'row': row,
                'types': ModelInfo.MODEL_TYPE_CHOICES,
                'form_file_path': form_file_path,
                'form_vectorizer_path': form_vectorizer_path,
                'form_config_json': form_config_json,
                'model_type_locked': model_type_locked,
                'fusion_strategy_edit': is_fusion_strategy,
            },
        )

    def post(self, request, model_id=None):
        if not model_id:
            messages.warning(request, '暂不支持新增模型记录')
            return redirect('model_list')
        row = ModelInfo.objects.filter(id=model_id).first()
        if not row:
            messages.warning(request, '未找到该模型记录')
            return redirect('model_list')
        type_locked = _model_edit_model_type_locked(row)
        is_fusion_strategy = is_canonical_fusion_registry_row(row)

        name_in = request.POST.get('name', '').strip()
        type_in = request.POST.get('model_type', '').strip()
        ver_in = request.POST.get('version', '').strip()

        name = name_in or row.name
        if type_locked:
            model_type = row.model_type
        else:
            model_type = type_in or row.model_type
        if is_fusion_strategy:
            version = row.version
        else:
            version = ver_in or row.version or 'v1'

        fp_in = request.POST.get('file_path', '').strip()
        vp_in = request.POST.get('vectorizer_path', '').strip()
        cfg_raw = request.POST.get('config_json', '').strip()

        PredictService.ensure_default_runtime_config()
        try:
            if is_fusion_strategy:
                extras = _apply_model_edit_config_json_post(cfg_raw, skip_path_keys=False, write_all_keys=True)
            elif row.model_type in SOURCE_ALGO_TYPES:
                _write_algo_paths_to_system_config(row.model_type, fp_in, vp_in)
                extras = _apply_model_edit_config_json_post(cfg_raw, skip_path_keys=True, write_all_keys=False)
            else:
                extras = _apply_model_edit_config_json_post(cfg_raw, skip_path_keys=True, write_all_keys=False)
        except ValueError as e:
            messages.error(request, str(e))
            return redirect(reverse('model_edit', args=[model_id]))

        config_json_stored = json.dumps(extras, ensure_ascii=False) if extras else '{}'

        if is_fusion_strategy:
            fp_store, vp_store = None, None
        elif row.model_type in SOURCE_ALGO_TYPES:
            fp_store = _norm_model_rel_path(fp_in) or None
            vp_store = _norm_model_rel_path(vp_in) or None
        else:
            fp_store = fp_in.strip() or None
            vp_store = vp_in.strip() or None

        payload = {
            'name': name,
            'model_type': model_type,
            'version': version,
            'file_path': fp_store,
            'vectorizer_path': vp_store,
            'description': request.POST.get('description', '').strip(),
            'config_json': config_json_stored,
            'metrics_json': request.POST.get('metrics_json', '').strip() or '{}',
            'status': request.POST.get('status', 'ready'),
            'listed_for_users': request.POST.get('listed_for_users') == 'on',
        }
        if is_primary_fusion_system_sync_row(row):
            payload.pop('metrics_json', None)

        ModelInfo.objects.filter(id=model_id).update(**payload)
        sync_modelinfo_paths_from_fusion_config()
        row_saved = ModelInfo.objects.filter(id=model_id).first()
        if (
            row_saved
            and row_saved.model_type == FUSION_STRATEGY_MODEL_TYPE
            and not is_primary_fusion_system_sync_row(row_saved)
        ):
            cfg_snap = PredictService.get_runtime_config(row_saved)
            snap_d = _build_fusion_snapshot_dict_for_cfg(cfg_snap)
            try:
                root = json.loads(row_saved.metrics_json or '{}')
            except Exception:
                root = {}
            if not isinstance(root, dict):
                root = {}
            root['fusion_snapshot'] = snap_d
            ModelInfo.objects.filter(id=model_id).update(
                metrics_json=json.dumps(root, ensure_ascii=False, indent=2)
            )
        _touch_autosave_preset()
        messages.info(
            request,
            '已保存：模型参数已写回；「融合策略」主登记已同步全局快照；其他名称的融合登记已按当前参数刷新 fusion_snapshot。',
        )
        return redirect('model_list')


@method_decorator(check_admin_access, name='dispatch')
class RuntimeConfigView(View):
    def get(self, request):
        userinfo = _get_login_userinfo(request)
        PredictService.ensure_default_runtime_config()
        rows = list(SystemConfig.objects.all().order_by('key'))
        sections = _build_fusion_config_sections(rows)
        active = ModelInfo.objects.filter(is_active=True, status='ready').order_by('-created_at').first()
        ctx = {
            'userinfo': userinfo,
            'fusion_sections': sections,
            'current_active': active,
            **_fusion_presets_for_template(),
        }
        return render(request, 'model/runtime_config.html', ctx)

    def post(self, request):
        action = (request.POST.get('action') or 'save').strip()

        if action in ('apply_preset', 'save_named_preset', 'delete_preset'):
            messages.warning(request, '配置快照功能已关闭。')
            return redirect('runtime_config')

        PredictService.ensure_default_runtime_config()
        bool_keys = _fusion_bool_keys()
        for row in SystemConfig.objects.all():
            fk = 'cfg_{}'.format(row.key)
            if row.key in bool_keys:
                row.value = 'True' if request.POST.get(fk) == '1' else 'False'
                row.save(update_fields=['value', 'updated_at'])
            elif fk in request.POST:
                val = request.POST.get(fk, '').strip()
                if val != '':
                    row.value = val[:255]
                    row.save(update_fields=['value', 'updated_at'])
        sync_modelinfo_paths_from_fusion_config()
        auto_name = _create_autonamed_fusion_registration_after_runtime_save()
        messages.info(
            request,
            '已保存。新预测将按当前设置执行；主登记「融合策略」已同步；'
            '已自动新增策略登记「{}」（可在模型管理中编辑或删除）。'.format(auto_name),
        )
        return redirect('runtime_config')


@method_decorator(check_admin_access, name='dispatch')
class ActiveModelView(View):
    def get(self, request):
        userinfo = _get_login_userinfo(request)
        active = ModelInfo.objects.filter(is_active=True, status='ready').order_by('-created_at').first()
        runtime_cfg = PredictService.get_runtime_config(active) if active else PredictService.get_runtime_config(None)
        summary_sections = _runtime_summary_sections(runtime_cfg)
        ctx = {
            'userinfo': userinfo,
            'active': active,
            'runtime_cfg': runtime_cfg,
            'summary_sections': summary_sections,
            **_fusion_presets_for_template(),
        }
        return render(request, 'model/active_model.html', ctx)


@method_decorator(check_admin_access, name='dispatch')
class ModelEvaluationView(View):
    def get(self, request):
        userinfo = _get_login_userinfo(request)
        _ensure_evaluation_modelinfo()
        model_id = request.GET.get('model_id')
        filter_model_id = int(model_id) if model_id and str(model_id).isdigit() else None
        rows = ModelEvaluation.objects.select_related('model_info').all().order_by('-evaluated_at')
        if filter_model_id is not None:
            rows = rows.filter(model_info_id=filter_model_id)
        models = ModelInfo.objects.all().order_by('-created_at')
        return render(
            request,
            'model/evaluation_list.html',
            {
                'userinfo': userinfo,
                'rows': rows,
                'models': models,
                'filter_model_id': filter_model_id,
            },
        )


@method_decorator(check_admin_access, name='dispatch')
class AlgorithmCompareView(View):
    def get(self, request):
        userinfo = _get_login_userinfo(request)
        return render(
            request,
            'model/algorithm_compare.html',
            {
                'userinfo': userinfo,
                'compare_rows': build_algorithm_compare_rows(),
                'fusion_compare_rows': build_fusion_strategy_compare_rows(),
            },
        )


@method_decorator(check_admin_access, name='dispatch')
class AlgorithmExperimentRecordListView(View):
    def get(self, request):
        userinfo = _get_login_userinfo(request)
        algorithm_type = (request.GET.get('algorithm_type') or '').strip()
        dataset_version = (request.GET.get('dataset_version') or '').strip()
        rows = AlgorithmExperimentRecord.objects.all().order_by('-created_at', '-id')
        if algorithm_type:
            rows = rows.filter(algorithm_type=algorithm_type)
        if dataset_version:
            rows = rows.filter(dataset_version__icontains=dataset_version)
        return render(
            request,
            'model/experiment_record_list.html',
            {
                'userinfo': userinfo,
                'rows': rows,
                'algorithm_type': algorithm_type,
                'dataset_version': dataset_version,
                'algo_choices': AlgorithmExperimentRecord.ALGO_TYPE_CHOICES,
            },
        )

    def post(self, request):
        action = request.POST.get('action')
        record_id = request.POST.get('record_id')
        params = {}
        if (request.POST.get('filter_algorithm_type') or '').strip():
            params['algorithm_type'] = request.POST.get('filter_algorithm_type').strip()
        if (request.POST.get('filter_dataset_version') or '').strip():
            params['dataset_version'] = request.POST.get('filter_dataset_version').strip()
        list_url = reverse('algorithm_experiment_record_list')
        if params:
            list_url = '{}?{}'.format(list_url, urlencode(params))

        if action == 'delete' and record_id:
            row = get_object_or_404(AlgorithmExperimentRecord, id=record_id)
            label = row.name
            rid = row.id
            row.delete()
            messages.success(request, '已删除实验记录：{}（ID {}）'.format(label, rid))
        else:
            messages.warning(request, '无效操作')
        return redirect(list_url)


@method_decorator(check_admin_access, name='dispatch')
class AlgorithmExperimentRecordFormView(View):
    def get(self, request, record_id=None):
        userinfo = _get_login_userinfo(request)
        row = AlgorithmExperimentRecord.objects.filter(id=record_id).first() if record_id else None
        return render(
            request,
            'model/experiment_record_form.html',
            {
                'userinfo': userinfo,
                'row': row,
                'algo_choices': AlgorithmExperimentRecord.ALGO_TYPE_CHOICES,
                'train_sample_choices': AlgorithmExperimentRecord.TRAIN_SAMPLE_CHOICES,
                'sklearn_mf_choices': AlgorithmExperimentRecord.SKLEARN_MAX_FEATURES_CHOICES,
            },
        )

    def post(self, request, record_id=None):
        row = AlgorithmExperimentRecord.objects.filter(id=record_id).first() if record_id else None
        valid_sample = {c[0] for c in AlgorithmExperimentRecord.TRAIN_SAMPLE_CHOICES}
        valid_mf = {c[0] for c in AlgorithmExperimentRecord.SKLEARN_MAX_FEATURES_CHOICES}
        train_sample_scale = (request.POST.get('train_sample_scale') or 'full').strip()
        sklearn_max_features = (request.POST.get('sklearn_max_features') or '8000').strip()
        if train_sample_scale not in valid_sample:
            train_sample_scale = 'full'
        if sklearn_max_features not in valid_mf:
            sklearn_max_features = '8000'

        payload = {
            'name': (request.POST.get('name') or '').strip(),
            'dataset_version': (request.POST.get('dataset_version') or '').strip(),
            'algorithm_type': (request.POST.get('algorithm_type') or '').strip(),
            'train_sample_scale': train_sample_scale,
            'sklearn_max_features': sklearn_max_features,
            'hyperparams_json': (request.POST.get('hyperparams_json') or '').strip() or '{}',
            'remark': (request.POST.get('remark') or '').strip()[:255],
        }
        if not payload['name'] or not payload['dataset_version'] or not payload['algorithm_type']:
            messages.error(request, '实验名称 / 数据版本 / 算法类型为必填项')
            return redirect('algorithm_experiment_record_list')

        def _float_or_default(raw, default=0.0):
            s = (raw or '').strip()
            if s == '':
                return default
            try:
                return float(s)
            except (TypeError, ValueError):
                return default

        payload['training_time_sec'] = _float_or_default(request.POST.get('training_time_sec'), None)
        payload['accuracy'] = _float_or_default(request.POST.get('accuracy'), 0.0)
        payload['precision'] = _float_or_default(request.POST.get('precision'), 0.0)
        payload['recall'] = _float_or_default(request.POST.get('recall'), 0.0)
        payload['f1_score'] = _float_or_default(request.POST.get('f1_score'), 0.0)
        payload['auc'] = _float_or_default(request.POST.get('auc'), None)
        if payload['training_time_sec'] is not None and payload['training_time_sec'] < 0:
            payload['training_time_sec'] = None
        if payload['auc'] is not None and payload['auc'] < 0:
            payload['auc'] = None

        try:
            json.loads(payload['hyperparams_json'])
        except Exception:
            messages.error(request, '超参数 JSON 格式不合法')
            if row is None:
                return redirect('algorithm_experiment_record_add')
            return redirect('algorithm_experiment_record_edit', record_id=row.id)

        if row is None:
            AlgorithmExperimentRecord.objects.create(**payload)
            messages.success(request, '实验记录已创建')
        else:
            for k, v in payload.items():
                setattr(row, k, v)
            row.save()
            messages.success(request, '实验记录已更新')
        return redirect('algorithm_experiment_record_list')


@method_decorator(check_admin_access, name='dispatch')
class ModelSelfCheckView(View):
    def get(self, request):
        userinfo = _get_login_userinfo(request)
        mode = request.GET.get('mode', 'active')
        if mode == 'all':
            reports = ModelHealthService.check_all_models()
        else:
            reports = [ModelHealthService.check_active_model()]
        summary = 'error' if any(r.get('summary') == 'error' for r in reports) else ('warning' if any(r.get('summary') == 'warning' for r in reports) else 'normal')
        has_error = summary == 'error'
        ModelSelfCheckRecord.objects.create(
            mode=mode,
            summary=summary,
            has_error=has_error,
            detail_json=json.dumps(reports, ensure_ascii=False),
        )
        return render(request, 'model/self_check.html', {'userinfo': userinfo, 'reports': reports, 'mode': mode})


@method_decorator(check_admin_access, name='dispatch')
class ModelSelfCheckHistoryView(View):
    def get(self, request):
        userinfo = _get_login_userinfo(request)
        rows = ModelSelfCheckRecord.objects.all().order_by('-created_at')
        export = request.GET.get('export', '')
        if export in ('csv', 'json'):
            return self._export(request, rows, export)
        return render(request, 'model/self_check_history.html', {'userinfo': userinfo, 'rows': rows})

    def _export(self, request, rows, export):
        userinfo = _get_login_userinfo(request)
        params = dict(request.GET)
        if 'export' in params:
            params.pop('export')
        normalized = {k: v[0] if isinstance(v, list) and v else v for k, v in params.items()}
        ExportLog.objects.create(
            export_type='self_check_history',
            export_format=export,
            exporter=userinfo,
            filter_json=json.dumps(normalized, ensure_ascii=False),
            export_count=rows.count(),
        )
        if export == 'json':
            data = [
                {'id': r.id, 'mode': r.mode, 'summary': r.summary, 'has_error': r.has_error, 'detail_json': r.detail_json, 'created_at': r.created_at.strftime('%Y-%m-%d %H:%M:%S')}
                for r in rows
            ]
            response = HttpResponse(json.dumps(data, ensure_ascii=False, indent=2), content_type='application/json')
            response['Content-Disposition'] = 'attachment; filename="model_selfcheck_history.json"'
            return response

        response = HttpResponse(content_type='text/csv; charset=utf-8-sig')
        response['Content-Disposition'] = 'attachment; filename="model_selfcheck_history.csv"'
        writer = csv.writer(response)
        writer.writerow(['ID', '检测模式', '摘要', '是否错误', '时间'])
        for r in rows:
            writer.writerow([r.id, r.mode, r.summary, '是' if r.has_error else '否', r.created_at.strftime('%Y-%m-%d %H:%M:%S')])
        return response


@method_decorator(check_admin_access, name='dispatch')
class ModelSelfCheckHistoryDetailView(View):
    def get(self, request, record_id):
        userinfo = _get_login_userinfo(request)
        row = get_object_or_404(ModelSelfCheckRecord, id=record_id)
        detail = []
        try:
            detail = json.loads(row.detail_json)
        except Exception:
            detail = [{'model': None, 'summary': 'error', 'items': [{'level': 'error', 'check': 'detail_json', 'message': row.detail_json}]}]
        return render(request, 'model/self_check_detail.html', {'userinfo': userinfo, 'row': row, 'detail': detail})
