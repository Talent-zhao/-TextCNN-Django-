# -*- coding: utf-8 -*-
"""
训练产物元数据：写入 model/model_training_registry.json，供「模型管理」页展示训练次数、召回率、损失等。
由 scripts/train_single_sklearn.py、train_char_torch.py 在成功结束时调用。
"""
import json
import os
from datetime import datetime

REGISTRY_FILENAME = 'model_training_registry.json'


def registry_abs_path(base_dir):
    return os.path.join(base_dir, 'model', REGISTRY_FILENAME)


def load_registry(base_dir):
    p = registry_abs_path(base_dir)
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_registry(base_dir, data):
    p = registry_abs_path(base_dir)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _norm_rel(rel):
    return (rel or '').replace('\\', '/').lstrip('/')


def record_training_session(
    base_dir, rel_paths, csv_rel, algo, recall_macro, last_loss, train_count_increment=1
):
    """
    在一次训练成功后，为若干相对路径（如 model/svm/model_svm.pkl）累加 train_count、更新指标。

    train_count_increment: 写入注册表时在原 train_count 上累加的步长；0 表示不增加次数，仍更新召回/损失等。
    recall_macro: 本次验证 macro avg recall，或与历史取 max 写入 best_recall_macro
    last_loss: 本次训练损失（Torch 为末轮 train_loss）；sklearn 可传 None
    """
    rel_paths = [_norm_rel(x) for x in rel_paths if x]
    if not rel_paths:
        return
    reg = load_registry(base_dir)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    csv_rel = _norm_rel(csv_rel) if csv_rel else ''

    try:
        r = float(recall_macro) if recall_macro is not None else None
    except (TypeError, ValueError):
        r = None

    loss_val = None
    if last_loss is not None:
        try:
            loss_val = float(last_loss)
        except (TypeError, ValueError):
            loss_val = None

    try:
        inc = int(train_count_increment)
    except (TypeError, ValueError):
        inc = 1
    if inc < 0:
        inc = 0
    if inc > 999:
        inc = 999

    for rel in rel_paths:
        cur = reg.get(rel) or {}
        prev = int(cur.get('train_count', 0))
        n = prev + inc if inc > 0 else prev
        best = cur.get('best_recall_macro')
        if r is not None:
            best = r if best is None else max(float(best), r)
        reg[rel] = {
            'train_count': n,
            'best_recall_macro': best,
            'last_recall_macro': r,
            'last_loss': loss_val,
            'last_train_at': now,
            'last_csv': csv_rel,
            'last_algo': (algo or '')[:64],
        }
    save_registry(base_dir, reg)


def set_registry_train_count(base_dir, rel, count):
    """手动覆盖注册表中的训练次数（其余字段保留；无记录则新建仅含 train_count）。"""
    rel = _norm_rel(rel)
    if count < 0 or count > 999999:
        raise ValueError('训练次数须在 0～999999 之间')
    reg = load_registry(base_dir)
    cur = dict(reg.get(rel) or {})
    cur['train_count'] = int(count)
    reg[rel] = cur
    save_registry(base_dir, reg)


def csv_arg_to_rel(csv_arg, root):
    """命令行 CSV 路径转为相对项目根的路径（便于页面展示）。"""
    if not csv_arg:
        return ''
    raw = os.path.normpath(str(csv_arg))
    root = os.path.normpath(root)
    if os.path.isabs(raw):
        try:
            rel = os.path.relpath(raw, root)
            if not rel.startswith('..'):
                return rel.replace('\\', '/')
        except ValueError:
            pass
        return raw.replace('\\', '/')
    return raw.replace('\\', '/')
