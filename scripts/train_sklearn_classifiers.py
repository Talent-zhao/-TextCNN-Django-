# -*- coding: utf-8 -*-
"""
基于与线上一致的 jieba+停用词预处理 + TF-IDF，训练 KNN / 随机森林 / 决策树 / 逻辑回归，
并保存为 joblib，按算法写入 model/<algo>/（与 train_single_sklearn 目录约定一致）。

用法（项目根目录）:
  python scripts/train_sklearn_classifiers.py --csv datasets/depression_nlp/zh/oesd_keyword_binary/splits/train.csv

依赖: pandas, scikit-learn, jieba（与主项目一致）
"""
from __future__ import print_function

import argparse
import json
import os
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from algorithm.model_paths import NUM2NAME_REL
from algorithm.text_utils import preprocess_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='含列 text, label 的 CSV')
    parser.add_argument(
        '--out-dir',
        default='',
        help='项目内 model 根目录；留空为 <项目>/model',
    )
    parser.add_argument('--max-features', type=int, default=8000)
    parser.add_argument(
        '--reuse-vectorizer',
        default='',
        help='若指定且文件存在，则用该 TF-IDF 只做 transform；否则重新 fit。默认尝试 model/svm/tfidfVectorizer.pkl',
    )
    args = parser.parse_args()

    model_root = (args.out_dir or '').strip() or os.path.join(ROOT, 'model')

    df = pd.read_csv(args.csv, encoding='utf-8-sig')
    if 'text' not in df.columns or 'label' not in df.columns:
        print('CSV 需要 text, label 列')
        sys.exit(1)

    texts = [preprocess_pipeline(t) for t in df['text'].astype(str)]
    le = LabelEncoder()
    y = le.fit_transform(df['label'].astype(str))

    os.makedirs(model_root, exist_ok=True)
    svm_dir = os.path.join(model_root, 'svm')
    reuse = args.reuse_vectorizer or os.path.join(svm_dir, 'tfidfVectorizer.pkl')
    if os.path.isfile(reuse):
        tv = joblib.load(reuse)
        X = tv.transform(texts)
        print('使用已有向量器 transform:', reuse)
    else:
        tv = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2))
        X = tv.fit_transform(texts)
        os.makedirs(svm_dir, exist_ok=True)
        vec_path = os.path.join(svm_dir, 'tfidfVectorizer_sklearn_multi.pkl')
        joblib.dump(tv, vec_path)
        print('已保存新向量器（SVM 目录）:', vec_path)

    models = {
        'model_knn.pkl': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'model_rf.pkl': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'model_dt.pkl': DecisionTreeClassifier(random_state=42, max_depth=20),
        'model_lr.pkl': LogisticRegression(max_iter=200, random_state=42, n_jobs=-1),
    }
    role_by_file = {
        'model_knn.pkl': 'knn',
        'model_rf.pkl': 'rf',
        'model_dt.pkl': 'dt',
        'model_lr.pkl': 'lr',
    }
    for fname, clf in models.items():
        clf.fit(X, y)
        role = role_by_file.get(fname)
        sub = os.path.join(model_root, role)
        os.makedirs(sub, exist_ok=True)
        path = os.path.join(sub, fname)
        joblib.dump(clf, path)
        print('已保存', path)
        if role:
            rv = os.path.join(sub, 'tfidfVectorizer_{}.pkl'.format(role))
            joblib.dump(tv, rv)
            print('  同步向量器', rv)

    svm_dir = os.path.join(model_root, 'svm')
    os.makedirs(svm_dir, exist_ok=True)
    common_vec = os.path.join(svm_dir, 'tfidfVectorizer.pkl')
    joblib.dump(tv, common_vec)
    print('已写入共用 TF-IDF（SVM 目录）:', common_vec)

    num2name = {str(i): str(le.classes_[i]) for i in range(len(le.classes_))}
    n2p = os.path.join(ROOT, *NUM2NAME_REL.split('/'))
    with open(n2p, 'w', encoding='utf-8') as f:
        json.dump(num2name, f, ensure_ascii=False, indent=2)
    print('标签映射已写入', n2p)
    n2p_ext = os.path.join(model_root, 'num2name_sklearn_extended.json')
    with open(n2p_ext, 'w', encoding='utf-8') as f:
        json.dump(num2name, f, ensure_ascii=False, indent=2)
    print('副本（便于对照）:', n2p_ext)
    print('完成。请在后台「融合配置」中指向 model/<algo>/ 下对应 pkl。')


if __name__ == '__main__':
    main()
