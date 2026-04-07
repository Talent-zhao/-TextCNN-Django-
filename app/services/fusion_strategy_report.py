# -*- coding: utf-8 -*-
"""
融合策略对比宽表：基于验证集与基分类器 predict_proba / softmax 做加权概率融合，
计算分类指标、Log-loss、Brier、AUC(OvR macro)、权重熵、留一 F1 降幅、自助法 F1 标准差、推理耗时与模型体积等。
"""
import csv
import math
import os
import time

import numpy as np
from django.conf import settings
from django.core.cache import cache

def _progress_interval():
    raw = (os.environ.get('FUSION_COMPARE_PROGRESS_EVERY') or '100').strip()
    try:
        n = int(raw)
    except ValueError:
        n = 100
    return max(1, n)


def _console_log(msg):
    # 算法对比计算过程输出到 runserver 控制台，便于观测进度与耗时。
    print('[fusion-compare] {}'.format(msg), flush=True)

from algorithm.fusion import finite_float
from algorithm.text_utils import load_num2name
from app.models import ModelEvaluation
from app.services.fusion_eval_service import DEFAULT_FUSION_EVAL_CSV_REL, _max_samples, _runtime_cfg_signature
from app.services.predict_service import PredictService

FUSION_PROB_ROLES = ('svm', 'knn', 'rf', 'dt', 'lr', 'textcnn', 'textrcnn')


def participating_raw_weights_seven(cfg):
    """仅统计「当前配置里已启用且 weight>0」的基模型原始权重；未启用为 0。"""
    row = []
    for r in FUSION_PROB_ROLES:
        if not cfg.get('enable_{}'.format(r), False):
            row.append(0.0)
            continue
        w = finite_float(cfg.get('weight_{}'.format(r), 0))
        row.append(w if w > 0 else 0.0)
    return row


def display_normalized_weights_seven(participating_raw):
    """在参与融合的基模型之间归一化（和为 1）；全无则全 0。"""
    s = sum(participating_raw)
    if s <= 1e-15:
        return [0.0] * len(FUSION_PROB_ROLES)
    return [x / s for x in participating_raw]


def entropy_distribution(p_list):
    """H = -sum p_i log2(p_i)，p_i 为已归一化分布（可与表格权重列一致）。"""
    h = 0.0
    for p in p_list:
        pf = float(p)
        if pf > 1e-15:
            h -= pf * math.log(pf, 2.0)
    return h


def _fmt4(x):
    if x is None:
        return '—'
    try:
        return '{:.4f}'.format(float(x))
    except (TypeError, ValueError):
        return '—'


def _fmt4pct_mean_drop(x):
    """留一 F1 降幅（百分比）。"""
    if x is None:
        return '—'
    try:
        return '{:.4f}'.format(float(x))
    except (TypeError, ValueError):
        return '—'


def _fmt_w(x):
    if x is None:
        return '—'
    try:
        return '{:.4f}'.format(float(x))
    except (TypeError, ValueError):
        return '—'


def _fms(x):
    if x is None:
        return '—'
    try:
        return '{:.2f}'.format(float(x))
    except (TypeError, ValueError):
        return '—'


def fused_probability_distribution(result_map, cfg, classes):
    """
    对含 proba_by_label 的基模型按权重做加权平均（不含规则）。
    是否参与融合仅以 cfg 的 enable_* + weight_* + 子模型 available 为准，
    便于留一评测时复用同一份 result_map、只改 cfg 权重。
    """
    w_sum = 0.0
    acc = {str(c): 0.0 for c in classes}
    for role in FUSION_PROB_ROLES:
        if not cfg.get('enable_{}'.format(role), False):
            continue
        r = result_map.get(role) or {}
        if not r.get('available'):
            continue
        wkey = 'weight_{}'.format(role)
        w = finite_float(cfg.get(wkey, 0))
        if w <= 0:
            continue
        pb = r.get('proba_by_label') or {}
        if not pb:
            continue
        w_sum += w
        for c in classes:
            k = str(c)
            v = pb.get(k)
            if v is None:
                v = pb.get(c, 0.0)
            acc[k] += w * float(v)
    if w_sum <= 1e-15:
        return None
    out = {str(c): acc[str(c)] / w_sum for c in classes}
    s2 = sum(out.values())
    if s2 > 1e-15:
        out = {k: v / s2 for k, v in out.items()}
    return out


def _forward_result_maps(texts, model_info, cfg):
    """对每个样本做一次各基模型前向，供主评测与留一只重算融合层。"""
    rms = []
    infer_ms = []
    total = len(texts)
    step = _progress_interval()
    for i, t in enumerate(texts, 1):
        t0 = time.perf_counter()
        rms.append(PredictService.build_result_map(t, model_info, cfg))
        infer_ms.append((time.perf_counter() - t0) * 1000.0)
        if i == 1 or i == total or (i % step == 0):
            _console_log(
                'forward progress {}/{} ({:.1f}%)'.format(i, total, 100.0 * i / max(total, 1))
            )
    return rms, infer_ms


def _predictions_from_maps(result_maps, labels, classes, cfg):
    """仅在缓存的 result_map 上用当前 cfg 做概率融合与 argmax。"""
    y_true, y_pred, prob_rows = [], [], []
    for rm, gold in zip(result_maps, labels):
        fp = fused_probability_distribution(rm, cfg, classes)
        if fp is None:
            continue
        y_true.append(str(gold).strip())
        y_pred.append(max(fp.keys(), key=lambda k: fp[k]))
        prob_rows.append(fp)
    return y_true, y_pred, prob_rows


def _multiclass_brier(y_true_str, prob_rows, classes):
    idx = {str(c): i for i, c in enumerate(classes)}
    n, k = len(y_true_str), len(classes)
    if n == 0 or k == 0:
        return None
    P = np.zeros((n, k), dtype=float)
    Y = np.zeros((n, k), dtype=float)
    for i, row in enumerate(prob_rows):
        for j, c in enumerate(classes):
            P[i, j] = float(row.get(str(c), 0.0))
        t = str(y_true_str[i]).strip()
        j = idx.get(t)
        if j is not None:
            Y[i, j] = 1.0
    return float(np.mean(np.sum((Y - P) ** 2, axis=1)))


def _collect_model_size_mb(model_info, cfg):
    total = 0.0
    for role in FUSION_PROB_ROLES:
        if not cfg.get('enable_{}'.format(role), False):
            continue
        paths = []
        if role in ('svm', 'knn', 'rf', 'dt', 'lr'):
            paths.append(
                PredictService._resolve_clf_path(model_info, cfg, role),
            )
            paths.append(
                PredictService._resolve_vec_path(model_info, cfg, role),
            )
        elif role == 'textcnn':
            vp = cfg.get('textcnn_vocab_path')
            wp = cfg.get('textcnn_weight_path')
            paths.extend([vp, wp])
        elif role == 'textrcnn':
            vp = cfg.get('textrcnn_vocab_path') or cfg.get('textcnn_vocab_path')
            wp = cfg.get('textrcnn_weight_path')
            paths.extend([vp, wp])
        for rel in paths:
            if not rel:
                continue
            full = os.path.join(settings.BASE_DIR, str(rel).replace('/', os.sep))
            if os.path.isfile(full):
                total += os.path.getsize(full) / (1024.0 * 1024.0)
    return total


def _sum_registered_training_sec(cfg):
    tot = 0.0
    any_ = False
    for role in FUSION_PROB_ROLES:
        if not cfg.get('enable_{}'.format(role), False):
            continue
        ev = (
            ModelEvaluation.objects.filter(model_info__model_type=role)
            .order_by('-evaluated_at')
            .first()
        )
        if ev and ev.training_time_sec is not None:
            tot += float(ev.training_time_sec)
            any_ = True
    return tot if any_ else None


def _union_class_names(labels, cfg):
    s = {str(x).strip() for x in labels if str(x).strip()}
    rel = (cfg.get('num2name_path') or 'model/num2name.json').strip()
    fulln = os.path.join(settings.BASE_DIR, rel.replace('/', os.sep))
    nm = load_num2name(fulln)
    for v in nm.values():
        s.add(str(v))
    return sorted(s)


def compute_fusion_strategy_report(model_info, csv_rel=None, max_samples=None):
    t_all = time.perf_counter()
    out = {
        'ok': False,
        'error': '',
        'strategy_name': '{} / {}'.format(model_info.name, model_info.version),
    }
    _console_log('start strategy={} id={}'.format(out['strategy_name'], model_info.id))

    if os.environ.get('DEPRESSION_WEB_SKIP_FUSION_COMPARE_EVAL', '').strip().lower() in (
        '1',
        'true',
        'yes',
    ):
        out['error'] = '已跳过（DEPRESSION_WEB_SKIP_FUSION_COMPARE_EVAL）'
        _console_log('skip by env: {}'.format(out['error']))
        return out

    csv_rel = (csv_rel or DEFAULT_FUSION_EVAL_CSV_REL).replace('\\', '/').lstrip('/')
    max_samples = max_samples if max_samples is not None else _max_samples()
    full = os.path.join(settings.BASE_DIR, csv_rel.replace('/', os.sep))
    if not os.path.isfile(full):
        out['error'] = '评估 CSV 不存在'
        return out

    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            f1_score,
            log_loss,
            roc_auc_score,
        )
    except Exception as e:
        out['error'] = 'sklearn: {}'.format(e)
        return out

    texts, labels = [], []
    with open(full, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            out['error'] = 'CSV 无表头'
            return out
        fields = {h.strip().lower(): h for h in reader.fieldnames}
        tk, lk = fields.get('text'), fields.get('label')
        if not tk or not lk:
            out['error'] = '需 text、label 列'
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
        out['error'] = '有效样本过少'
        _console_log('abort: {} n={}'.format(out['error'], len(texts)))
        return out

    _console_log('loaded samples={} csv={}'.format(len(texts), csv_rel))
    cfg0 = PredictService.get_runtime_config(model_info)
    classes = _union_class_names(labels, cfg0)
    _console_log('labels/classes count={} classes={}'.format(len(classes), ','.join(classes[:8])))

    t_fw = time.perf_counter()
    result_maps, infer_ms = _forward_result_maps(texts, model_info, cfg0)
    _console_log('forward done elapsed={:.2f}s'.format(time.perf_counter() - t_fw))
    y_true, y_pred, prob_rows = _predictions_from_maps(result_maps, labels, classes, cfg0)
    if len(y_true) < 10:
        out['error'] = '概率融合成功样本过少（基模型是否输出 proba_by_label？）'
        _console_log('abort: {} n={}'.format(out['error'], len(y_true)))
        return out

    acc = float(accuracy_score(y_true, y_pred))
    rep = classification_report(
        y_true,
        y_pred,
        labels=list(classes),
        output_dict=True,
        zero_division=0,
    )
    ma = rep.get('macro avg') or {}

    def _mf(k):
        try:
            return float(ma.get(k) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    prec, rec, f1 = _mf('precision'), _mf('recall'), _mf('f1-score')

    y_enc = np.array([classes.index(y) for y in y_true])
    P = np.clip(
        np.array([[pr.get(str(c), 0.0) for c in classes] for pr in prob_rows]),
        1e-15,
        1.0 - 1e-15,
    )
    P = P / P.sum(axis=1, keepdims=True)

    try:
        ll = float(log_loss(y_enc, P, labels=list(range(len(classes)))))
    except Exception:
        ll = None

    brier = _multiclass_brier(y_true, prob_rows, classes)

    auc = None
    try:
        if len(classes) == 2:
            pos_class = 'depression_related' if 'depression_related' in classes else classes[-1]
            pos_idx = classes.index(pos_class)
            y_bin = np.array([1 if str(y) == str(pos_class) else 0 for y in y_true])
            if len(set(y_bin)) == 2:
                auc = float(roc_auc_score(y_bin, P[:, pos_idx]))
        else:
            auc = float(
                roc_auc_score(
                    y_enc,
                    P,
                    multi_class='ovr',
                    average='macro',
                    labels=list(range(len(classes))),
                )
            )
    except Exception:
        auc = None

    part_raw = participating_raw_weights_seven(cfg0)
    norm_w = display_normalized_weights_seven(part_raw)
    ent = entropy_distribution(norm_w)

    for i, role in enumerate(FUSION_PROB_ROLES):
        out['w_{}'.format(role)] = norm_w[i]

    f1_full = f1
    loo_drops = []
    skip_loo = os.environ.get('FUSION_COMPARE_SKIP_LOO', '').strip().lower() in ('1', 'true', 'yes')
    if not skip_loo:
        _console_log('LOO start')
        for role in FUSION_PROB_ROLES:
            if finite_float(cfg0.get('weight_{}'.format(role), 0)) <= 0:
                continue
            if not cfg0.get('enable_{}'.format(role), False):
                continue
            cfg1 = dict(cfg0)
            cfg1['weight_{}'.format(role)] = 0.0
            cfg1['enable_{}'.format(role)] = False
            yt2, yp2, _ = _predictions_from_maps(result_maps, labels, classes, cfg1)
            if len(yt2) < 10:
                _console_log('LOO role={} skipped(valid<10)'.format(role))
                continue
            f2 = float(
                f1_score(yt2, yp2, labels=list(classes), average='macro', zero_division=0)
            )
            if f1_full > 1e-9:
                drop_pct = max(0.0, (f1_full - f2) / f1_full * 100.0)
                loo_drops.append(drop_pct)
                _console_log('LOO role={} f1_drop_pct={:.4f}'.format(role, drop_pct))
        _console_log('LOO done count={}'.format(len(loo_drops)))
    loo_mean = float(np.mean(loo_drops)) if loo_drops else None

    rng = np.random.RandomState(42)
    boot_f1 = []
    y_arr = np.array(y_true)
    p_arr = np.array(y_pred)
    n_ = len(y_true)
    for _ in range(int(os.environ.get('FUSION_COMPARE_BOOTSTRAP', '15'))):
        idx = rng.randint(0, n_, size=n_)
        boot_f1.append(
            float(
                f1_score(
                    y_arr[idx],
                    p_arr[idx],
                    labels=list(classes),
                    average='macro',
                    zero_division=0,
                )
            )
        )
    boot_std = float(np.std(boot_f1)) if boot_f1 else None

    ms_per = float(np.mean(infer_ms)) if infer_ms else None
    size_mb = _collect_model_size_mb(model_info, cfg0)
    train_s = _sum_registered_training_sec(cfg0)

    out.update(
        {
            'ok': True,
            'error': '',
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc': auc,
            'log_loss': ll,
            'brier': brier,
            'weight_entropy': ent,
            'loo_f1_drop_mean_pct': loo_mean,
            'bootstrap_f1_std': boot_std,
            'infer_ms_per_sample': ms_per,
            'model_size_mb': size_mb,
            'train_total_sec': train_s,
            'n_samples': len(y_true),
            'csv_rel': csv_rel,
        }
    )
    _console_log(
        'done strategy={} n={} acc={:.4f} f1={:.4f} auc={} elapsed={:.2f}s'.format(
            out['strategy_name'],
            out.get('n_samples', 0),
            acc,
            f1,
            '{:.4f}'.format(float(auc)) if auc is not None else '—',
            time.perf_counter() - t_all,
        )
    )
    return out


def _report_cache_key(model_info, csv_rel, max_samples):
    try:
        full = os.path.join(
            settings.BASE_DIR, (csv_rel or DEFAULT_FUSION_EVAL_CSV_REL).replace('/', os.sep)
        )
        mtime = os.path.getmtime(full) if os.path.isfile(full) else 0
    except OSError:
        mtime = 0
    sig = _runtime_cfg_signature(model_info)
    return 'fusion_strategy_report:v4:{}:{}:{}:{}:{}'.format(
        model_info.pk,
        sig,
        mtime,
        max_samples,
        csv_rel or DEFAULT_FUSION_EVAL_CSV_REL,
    )


def get_cached_fusion_strategy_report(model_info, csv_rel=None, max_samples=None):
    csv_rel = (csv_rel or DEFAULT_FUSION_EVAL_CSV_REL).replace('\\', '/').lstrip('/')
    max_samples = max_samples if max_samples is not None else _max_samples()
    key = _report_cache_key(model_info, csv_rel, max_samples)
    hit = cache.get(key)
    if hit is not None:
        _console_log('cache hit strategy={} id={}'.format(model_info.name, model_info.id))
        return hit
    _console_log('cache miss strategy={} id={}'.format(model_info.name, model_info.id))
    data = compute_fusion_strategy_report(model_info, csv_rel=csv_rel, max_samples=max_samples)
    ttl = 86400 if data.get('ok') else 120
    cache.set(key, data, ttl)
    return data


def report_to_table_row(rep):
    """转为模板行 dict（已格式化为字符串）。"""
    if not rep.get('ok'):
        row = {k: '—' for k in (
            'accuracy',
            'precision',
            'recall',
            'f1',
            'auc',
            'log_loss',
            'w_svm',
            'w_knn',
            'w_dt',
            'w_rf',
            'w_lr',
            'w_textcnn',
            'w_textrcnn',
            'weight_entropy',
            'loo_f1_drop',
            'bootstrap_f1_std',
            'brier',
            'infer_ms',
            'model_size_mb',
            'train_total_sec',
        )}
        row['strategy_name'] = rep.get('strategy_name', '—')
        row['error_note'] = (rep.get('error') or '')[:200]
        row['n_samples_note'] = ''
        row['csv_rel'] = ''
        return row

    return {
        'strategy_name': rep.get('strategy_name', '—'),
        'accuracy': _fmt4(rep.get('accuracy')),
        'precision': _fmt4(rep.get('precision')),
        'recall': _fmt4(rep.get('recall')),
        'f1': _fmt4(rep.get('f1')),
        'auc': _fmt4(rep.get('auc')),
        'log_loss': _fmt4(rep.get('log_loss')),
        'w_svm': _fmt_w(rep.get('w_svm')),
        'w_knn': _fmt_w(rep.get('w_knn')),
        'w_dt': _fmt_w(rep.get('w_dt')),
        'w_rf': _fmt_w(rep.get('w_rf')),
        'w_lr': _fmt_w(rep.get('w_lr')),
        'w_textcnn': _fmt_w(rep.get('w_textcnn')),
        'w_textrcnn': _fmt_w(rep.get('w_textrcnn')),
        'weight_entropy': _fmt4(rep.get('weight_entropy')),
        'loo_f1_drop': _fmt4pct_mean_drop(rep.get('loo_f1_drop_mean_pct')),
        'bootstrap_f1_std': _fmt4(rep.get('bootstrap_f1_std')),
        'brier': _fmt4(rep.get('brier')),
        'infer_ms': _fms(rep.get('infer_ms_per_sample')),
        'model_size_mb': _fms(rep.get('model_size_mb')),
        'train_total_sec': _fms(rep.get('train_total_sec')),
        'error_note': '',
        'n_samples_note': str(rep.get('n_samples', '')),
        'csv_rel': rep.get('csv_rel', ''),
    }
