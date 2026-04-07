# -*- coding: utf-8 -*-
"""
融合策略在「算法对比」页的离线指标估算：在标注 CSV 上逐条调用 PredictService.predict_text_line，
计算 Accuracy / macro P、R、F1；二分类时用融合 final_risk_score 近似算 AUC。

结果经 Django cache 按「配置签名 + CSV 修改时间 + 样本上限」缓存，避免每次打开页面全量重算。
"""
import csv
import hashlib
import json
import os

from django.conf import settings
from django.core.cache import cache

from app.services.predict_service import PredictService

DEFAULT_FUSION_EVAL_CSV_REL = '/'.join(
    (
        'datasets',
        'depression_nlp',
        'zh',
        'oesd_keyword_binary',
        'splits',
        'val.csv',
    )
)


def _max_samples():
    raw = (os.environ.get('FUSION_COMPARE_MAX_SAMPLES') or '800').strip()
    try:
        n = int(raw)
    except ValueError:
        n = 800
    return max(50, min(n, 10000))


def _runtime_cfg_signature(mi):
    cfg = PredictService.get_runtime_config(mi)
    keys = sorted(
        k
        for k in cfg
        if k.startswith(('enable_', 'weight_', 'threshold_'))
        or k.endswith('_model_path')
        or k.endswith('_weight_path')
        or k.endswith('_vocab_path')
        or k.startswith('tfidf_vectorizer_path')
        or k in ('tfidf_vectorizer_path', 'num2name_path')
    )
    sub = {k: cfg.get(k) for k in keys}
    cj = (mi.config_json or '').strip()
    mj = ''
    if mi and getattr(mi, 'model_type', None) == 'fusion':
        mj = (mi.metrics_json or '').strip()
    return hashlib.md5(
        json.dumps(
            {'cfg': sub, 'config_json': cj, 'metrics_json': mj},
            sort_keys=True,
            default=str,
        ).encode('utf-8')
    ).hexdigest()


def compute_fusion_eval_metrics(model_info, csv_rel=None, max_samples=None):
    """
    在 CSV（列 text, label）上评估当前融合登记 model_info。
    返回 dict: ok, error?, accuracy, precision, recall, f1, auc, n_samples, csv_rel
    """
    out = {'ok': False, 'error': ''}
    if os.environ.get('DEPRESSION_WEB_SKIP_FUSION_COMPARE_EVAL', '').strip().lower() in (
        '1',
        'true',
        'yes',
    ):
        out['error'] = '已跳过（DEPRESSION_WEB_SKIP_FUSION_COMPARE_EVAL）'
        return out

    csv_rel = (csv_rel or DEFAULT_FUSION_EVAL_CSV_REL).replace('\\', '/').lstrip('/')
    max_samples = max_samples if max_samples is not None else _max_samples()
    full = os.path.join(settings.BASE_DIR, csv_rel.replace('/', os.sep))
    if not os.path.isfile(full):
        out['error'] = '评估 CSV 不存在: {}'.format(csv_rel)
        return out

    texts, labels = [], []
    try:
        mtime = os.path.getmtime(full)
    except OSError:
        mtime = 0

    with open(full, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            out['error'] = 'CSV 无表头'
            return out
        fields = {h.strip().lower(): h for h in reader.fieldnames}
        tk = fields.get('text')
        lk = fields.get('label')
        if not tk or not lk:
            out['error'] = 'CSV 需包含 text、label 列'
            return out
        for i, row in enumerate(reader):
            if i >= max_samples:
                break
            t = (row.get(tk) or '').strip()
            lb = (row.get(lk) or '').strip()
            if not t:
                continue
            texts.append(t)
            labels.append(lb)

    if len(texts) < 10:
        out['error'] = '有效样本过少（{}）'.format(len(texts))
        return out

    y_true, y_pred, y_score = [], [], []
    for t, gold in zip(texts, labels):
        try:
            pr = PredictService.predict_text_line(t, model_info=model_info)
        except Exception:
            continue
        pred = pr.get('label')
        if pred is None:
            continue
        pred = str(pred).strip()
        y_true.append(gold)
        y_pred.append(pred)
        try:
            y_score.append(float(pr.get('final_score', 0.0)))
        except (TypeError, ValueError):
            y_score.append(0.0)

    if len(y_true) < 10:
        out['error'] = '成功预测过少（{}）'.format(len(y_true))
        return out

    try:
        from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
    except Exception as e:
        out['error'] = 'sklearn 不可用: {}'.format(e)
        return out

    acc = float(accuracy_score(y_true, y_pred))
    all_labels = sorted(set(y_true) | set(y_pred))
    rep = classification_report(
        y_true,
        y_pred,
        labels=all_labels,
        output_dict=True,
        zero_division=0,
    )
    ma = rep.get('macro avg') or {}

    def _f(x):
        try:
            return float(x) if x is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    prec = _f(ma.get('precision'))
    rec = _f(ma.get('recall'))
    f1 = _f(ma.get('f1-score'))

    auc = None
    if len(all_labels) == 2:
        pos = 'depression_related'
        if pos not in all_labels:
            pos = all_labels[-1]
        y_bin = [1 if str(g).strip() == pos else 0 for g in y_true]
        if len(set(y_bin)) == 2:
            try:
                auc = float(roc_auc_score(y_bin, y_score))
            except Exception:
                auc = None

    out.update(
        {
            'ok': True,
            'error': '',
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc': auc,
            'n_samples': len(y_true),
            'csv_rel': csv_rel,
            'csv_mtime': mtime,
        }
    )
    return out


def get_cached_fusion_eval_metrics(model_info, csv_rel=None, max_samples=None):
    csv_rel = (csv_rel or DEFAULT_FUSION_EVAL_CSV_REL).replace('\\', '/').lstrip('/')
    max_samples = max_samples if max_samples is not None else _max_samples()
    full = os.path.join(settings.BASE_DIR, csv_rel.replace('/', os.sep))
    try:
        mtime = os.path.getmtime(full) if os.path.isfile(full) else 0
    except OSError:
        mtime = 0
    sig = _runtime_cfg_signature(model_info)
    key = 'fusion_compare_eval:v1:{}:{}:{}:{}:{}'.format(
        model_info.pk,
        sig,
        mtime,
        max_samples,
        csv_rel,
    )
    hit = cache.get(key)
    if hit is not None:
        return hit
    data = compute_fusion_eval_metrics(model_info, csv_rel=csv_rel, max_samples=max_samples)
    ttl = 86400 if data.get('ok') else 120
    cache.set(key, data, ttl)
    return data


def overlay_computed_metrics_if_missing(cells, comp):
    """
    cells: _build_compare_row_cells 结果（已格式化为字符串）。
    comp: get_cached_fusion_eval_metrics 返回值。
    仅当对应单元格为「—」且 comp 成功时用离线测算填充。
    返回 (cells, computed_note: str)
    """
    if not comp or not comp.get('ok'):
        return cells, ''

    overlaid = False

    def _maybe(key, val):
        nonlocal overlaid
        if val is None:
            return
        if cells.get(key) != '—':
            return
        if key == 'train_time' or key == 'overfitting':
            return
        if key == 'auc' and val is None:
            return
        cells[key] = '{:.4f}'.format(float(val))
        overlaid = True

    _maybe('accuracy', comp.get('accuracy'))
    _maybe('precision', comp.get('precision'))
    _maybe('recall', comp.get('recall'))
    _maybe('f1', comp.get('f1'))
    if comp.get('auc') is not None:
        _maybe('auc', comp.get('auc'))

    note = ''
    if overlaid:
        note = '离线测算：{}（前 {} 条）'.format(comp.get('csv_rel', ''), comp.get('n_samples', ''))
    return cells, note
