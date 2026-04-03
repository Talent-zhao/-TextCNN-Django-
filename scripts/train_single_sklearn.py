# -*- coding: utf-8 -*-
"""
单独训练一个 TF-IDF + sklearn 分类器，默认写入 model/<algo>/（与 PredictService、model_paths 约定一致）。

用法（项目根）:
  python scripts/train_single_sklearn.py --csv datasets/depression_nlp/zh/oesd_keyword_binary/splits/train.csv --algo svm
  algo: svm | knn | rf | dt | lr

标签映射统一写入 model/num2name.json。
"""
from __future__ import print_function

import argparse
import json
import os
import sys
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from algorithm.model_paths import NUM2NAME_REL
from algorithm.model_training_registry import csv_arg_to_rel, record_training_session
from algorithm.text_utils import preprocess_pipeline
from algorithm.training_summary_console import print_experiment_form_footer

OUT_FILES = {
    'svm': 'model_svm.pkl',
    'knn': 'model_knn.pkl',
    'rf': 'model_rf.pkl',
    'dt': 'model_dt.pkl',
    'lr': 'model_lr.pkl',
}


def _experiment_sample_token_from_args(max_samples):
    try:
        n = int(max_samples)
    except (TypeError, ValueError):
        return 'full'
    if n <= 0:
        return 'full'
    mapping = {8000: '8000', 15000: '15000', 30000: '30000'}
    return mapping.get(n, 'full')


def _experiment_mf_token_from_args(max_features):
    try:
        n = int(max_features)
    except (TypeError, ValueError):
        return '8000'
    mapping = {4000: '4000', 8000: '8000', 12000: '12000'}
    return mapping.get(n, '8000')


def build_classifier(algo):
    algo = algo.lower()
    if algo == 'svm':
        return SVC(kernel='linear', probability=True, random_state=42)
    if algo == 'knn':
        return KNeighborsClassifier(n_neighbors=5, weights='distance')
    if algo == 'rf':
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    if algo == 'dt':
        return DecisionTreeClassifier(random_state=42, max_depth=25)
    if algo == 'lr':
        return LogisticRegression(
            max_iter=500, random_state=42, n_jobs=-1, solver='lbfgs', multi_class='auto'
        )
    raise ValueError('unknown algo: ' + algo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--algo', required=True, choices=list(OUT_FILES.keys()))
    parser.add_argument(
        '--out-dir',
        default='',
        help='默认 model/<algo>/；留空则按算法写入子目录',
    )
    parser.add_argument('--max-features', type=int, default=8000)
    parser.add_argument(
        '--max-samples',
        type=int,
        default=0,
        help='>0 时随机子采样该条数再训练（与深度学习页选项一致）',
    )
    parser.add_argument(
        '--registry-train-increment',
        type=int,
        default=1,
        help='成功写盘后注册表 train_count 累加步长；0 不增加但仍更新召回等指标',
    )
    args = parser.parse_args()
    out_dir = (args.out_dir or '').strip()
    if not out_dir:
        out_dir = os.path.join(ROOT, 'model', args.algo.lower())

    df = pd.read_csv(args.csv, encoding='utf-8-sig')
    if 'text' not in df.columns or 'label' not in df.columns:
        print('ERROR: CSV 需要列 text, label', flush=True)
        sys.exit(1)

    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)
        print('[0/4] 已按 --max-samples 子采样为', len(df), '条', flush=True)

    print('[1/4] 读取样本数:', len(df), flush=True)
    texts = [preprocess_pipeline(t) for t in df['text'].astype(str)]
    le = LabelEncoder()
    y = le.fit_transform(df['label'].astype(str))
    names = [str(x) for x in le.classes_]

    try:
        tr_txt, te_txt, y_tr, y_te = train_test_split(
            texts, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        tr_txt, te_txt, y_tr, y_te = train_test_split(
            texts, y, test_size=0.2, random_state=42, stratify=None
        )
    print('[2/4] 划分验证集: 训练', len(tr_txt), '验证', len(te_txt), flush=True)

    tv = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2))
    t_fit0 = time.perf_counter()
    X_tr = tv.fit_transform(tr_txt)
    X_te = tv.transform(te_txt)
    clf = build_classifier(args.algo)
    print('[3/4] 拟合', args.algo.upper(), '…', flush=True)
    clf.fit(X_tr, y_tr)
    fit_wall_sec = time.perf_counter() - t_fit0
    y_pred = clf.predict(X_te)
    y_pred_tr = clf.predict(X_tr)
    train_acc_eval = float(accuracy_score(y_tr, y_pred_tr))
    val_acc_eval = float(accuracy_score(y_te, y_pred))
    y_proba = None
    if hasattr(clf, 'predict_proba'):
        try:
            y_proba = clf.predict_proba(X_te)
        except Exception:
            y_proba = None
    n_cls = len(names)
    report = classification_report(
        y_te,
        y_pred,
        labels=list(range(n_cls)),
        target_names=names,
        digits=4,
        zero_division=0,
    )
    rep_dict = classification_report(
        y_te,
        y_pred,
        labels=list(range(n_cls)),
        target_names=names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    macro_recall = rep_dict.get('macro avg', {}).get('recall')
    try:
        macro_recall = float(macro_recall) if macro_recall is not None else None
    except (TypeError, ValueError):
        macro_recall = None
    print('--- 验证集分类报告（含召回率 recall、F1）---', flush=True)
    print(report, flush=True)

    print('[4/4] 全量重训并保存模型 …', flush=True)
    tv = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2))
    X = tv.fit_transform(texts)
    clf = build_classifier(args.algo)
    clf.fit(X, y)

    os.makedirs(out_dir, exist_ok=True)
    vec_path = os.path.join(out_dir, 'tfidfVectorizer.pkl')
    joblib.dump(tv, vec_path)
    role_vec = os.path.join(out_dir, 'tfidfVectorizer_{}.pkl'.format(args.algo))
    joblib.dump(tv, role_vec)
    mpath = os.path.join(out_dir, OUT_FILES[args.algo])
    joblib.dump(clf, mpath)

    num2name = {str(i): str(le.classes_[i]) for i in range(len(le.classes_))}
    os.makedirs(os.path.join(ROOT, 'model'), exist_ok=True)
    n2path = os.path.join(ROOT, *NUM2NAME_REL.split('/'))
    with open(n2path, 'w', encoding='utf-8') as f:
        json.dump(num2name, f, ensure_ascii=False, indent=2)

    rels = [
        os.path.relpath(mpath, ROOT).replace('\\', '/'),
        os.path.relpath(vec_path, ROOT).replace('\\', '/'),
        os.path.relpath(role_vec, ROOT).replace('\\', '/'),
        os.path.relpath(n2path, ROOT).replace('\\', '/'),
    ]
    record_training_session(
        ROOT,
        rels,
        csv_arg_to_rel(args.csv, ROOT),
        args.algo,
        macro_recall,
        None,
        train_count_increment=getattr(args, 'registry_train_increment', 1),
    )

    footer_acc = footer_p = footer_r = footer_f1 = footer_auc = None
    try:
        from app.training_evaluation import (
            infer_overfitting_from_train_val_acc,
            metrics_from_classification_report_dict,
            roc_auc_from_proba,
            save_evaluation_from_training,
        )

        acc, p, r, f1v = metrics_from_classification_report_dict(rep_dict, y_te, y_pred)
        auc_val = roc_auc_from_proba(y_te, y_proba) if y_proba is not None else None
        footer_acc, footer_p, footer_r, footer_f1, footer_auc = acc, p, r, f1v, auc_val
        if auc_val is not None:
            print('[eval] 验证集 ROC-AUC (OvR macro 或二分类): {:.4f}'.format(auc_val), flush=True)
        ofit = infer_overfitting_from_train_val_acc(train_acc_eval, val_acc_eval)
        if ofit is not None:
            print(
                '[eval] 训练准确率 {:.4f} vs 验证 {:.4f}，过拟合(训练-验证>6%): {}'.format(
                    train_acc_eval,
                    val_acc_eval,
                    '是' if ofit else '否',
                ),
                flush=True,
            )
        rel_model_save = os.path.relpath(mpath, ROOT).replace('\\', '/')
        rel_vec_save = os.path.relpath(vec_path, ROOT).replace('\\', '/')
        csv_rel_ex = csv_arg_to_rel(args.csv, ROOT)
        hp_ex = {
            'max_features': int(args.max_features),
            'max_samples': int(args.max_samples) if getattr(args, 'max_samples', 0) else None,
            'ngram_range': [1, 2],
            'model_path': rel_model_save,
            'vectorizer_path': rel_vec_save,
        }
        save_evaluation_from_training(
            args.algo,
            acc,
            p,
            r,
            f1v,
            auc=auc_val,
            training_time_sec=fit_wall_sec,
            is_overfitting=ofit,
            file_path=rel_model_save,
            vectorizer_path=rel_vec_save,
            experiment_csv_rel=csv_rel_ex,
            experiment_train_sample_scale=_experiment_sample_token_from_args(args.max_samples),
            experiment_sklearn_max_features=_experiment_mf_token_from_args(args.max_features),
            experiment_hyperparams_extra=hp_ex,
        )
    except Exception as ex:
        print('[eval_persist] 自动保存评估异常（可忽略）:', ex, flush=True)

    print('OK:', args.algo, flush=True)
    print('  out_dir:', out_dir, flush=True)
    print('  vectorizer:', vec_path, flush=True)
    print('  per-algo vectorizer:', role_vec, flush=True)
    print('  model:', mpath, flush=True)
    print('  num2name:', n2path, flush=True)
    print('  classes:', len(le.classes_), flush=True)

    hp_footer = {
        'algo': args.algo,
        'max_features': int(args.max_features),
        'max_samples': int(args.max_samples) if getattr(args, 'max_samples', 0) else None,
        'ngram_range': [1, 2],
        'csv': os.path.basename(str(args.csv)),
    }
    print_experiment_form_footer(
        hp_footer,
        fit_wall_sec,
        footer_auc,
        footer_acc,
        footer_p,
        footer_r,
        footer_f1,
    )


if __name__ == '__main__':
    main()
