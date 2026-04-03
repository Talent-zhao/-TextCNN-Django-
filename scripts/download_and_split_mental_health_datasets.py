# -*- coding: utf-8 -*-
"""
从 Hugging Face 下载公开心理健康/抑郁相关文本数据集，并划分 train / val / test（7:1.5:1.5，分层抽样）。

默认输出目录：项目根下 datasets/depression_nlp/en/（英文数据）；中文见 zh/oesd_keyword_binary/

用法（在项目根目录执行）:
    python scripts/download_and_split_mental_health_datasets.py

依赖：pandas、scikit-learn、requests（或仅用 urllib，下面已用 urllib 减少依赖）
"""
from __future__ import unicode_literals

import os
import sys
import urllib.request
import ssl

import pandas as pd
from sklearn.model_selection import train_test_split

# 项目根目录（含 manage.py 的目录）
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_NLP = os.path.join(ROOT, 'datasets', 'depression_nlp')
OUT_ROOT = os.path.join(DATA_NLP, 'en')

SOURCES = [
    {
        'name': 'mental_health_4class',
        'description': 'nayma11 多类：Anxiety / Depression / Suicidal / Normal',
        'files': [
            {
                'url': 'https://huggingface.co/datasets/nayma11/Mental-Health_Text-Classification_Dataset/resolve/main/mental_heath_unbanlanced.csv',
                'save_as': 'mental_heath_unbalanced.csv',
            },
        ],
        'loader': 'nayma11_unbalanced',
    },
    {
        'name': 'reddit_depression_binary',
        'description': 'Reddit 清洗二分类（is_depression）',
        'files': [
            {
                'url': 'https://huggingface.co/datasets/hugginglearners/reddit-depression-cleaned/resolve/main/depression_dataset_reddit_cleaned.csv',
                'save_as': 'depression_dataset_reddit_cleaned.csv',
            },
        ],
        'loader': 'reddit_cleaned',
    },
    {
        'name': 'depression_posts_binary',
        'description': 'joangaes 抑郁相关二分类（label 0/1）',
        'files': [
            {
                'url': 'https://huggingface.co/datasets/joangaes/depression/resolve/main/clean_encoded_df.csv',
                'save_as': 'clean_encoded_df.csv',
            },
        ],
        'loader': 'joangaes',
    },
]


def _download(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 1000:
        print('  已存在，跳过下载: {}'.format(dest_path))
        return
    print('  下载: {} -> {}'.format(url, dest_path))
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (research) Python'})
    with urllib.request.urlopen(req, context=ctx, timeout=300) as resp:
        data = resp.read()
    with open(dest_path, 'wb') as f:
        f.write(data)


def _split_save(df, text_col, label_col, split_dir, random_state=42):
    df = df[[text_col, label_col]].copy()
    df.rename(columns={text_col: 'text', label_col: 'label'}, inplace=True)
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'].str.len() > 0]
    df = df.dropna(subset=['label'])
    df.drop_duplicates(subset=['text'], inplace=True)

    X = df['text'].values
    y = df['label'].values

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_state, stratify=y
    )
    val_ratio = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=random_state, stratify=y_train_val
    )

    os.makedirs(split_dir, exist_ok=True)
    pd.DataFrame({'text': X_train, 'label': y_train}).to_csv(
        os.path.join(split_dir, 'train.csv'), index=False, encoding='utf-8-sig'
    )
    pd.DataFrame({'text': X_val, 'label': y_val}).to_csv(
        os.path.join(split_dir, 'val.csv'), index=False, encoding='utf-8-sig'
    )
    pd.DataFrame({'text': X_test, 'label': y_test}).to_csv(
        os.path.join(split_dir, 'test.csv'), index=False, encoding='utf-8-sig'
    )
    print('  划分完成: train={} val={} test={}'.format(len(X_train), len(X_val), len(X_test)))


def load_nayma11(path):
    df = pd.read_csv(path, encoding='utf-8')
    return df, 'text', 'status'


def load_reddit_cleaned(path):
    df = pd.read_csv(path, encoding='utf-8')
    df['label'] = df['is_depression'].map(
        lambda x: 'depression' if int(float(x)) == 1 else 'non_depression'
    )
    return df, 'clean_text', 'label'


def load_joangaes(path):
    df = pd.read_csv(path, encoding='utf-8')
    df['label'] = df['label'].map(lambda x: int(float(x)))
    return df, 'text', 'label'


LOADERS = {
    'nayma11_unbalanced': load_nayma11,
    'reddit_cleaned': load_reddit_cleaned,
    'joangaes': load_joangaes,
}


def main():
    if sys.version_info[0] < 3:
        print('请使用 Python 3 运行本脚本')
        sys.exit(1)

    print('输出根目录: {}'.format(OUT_ROOT))
    for src in SOURCES:
        name = src['name']
        raw_dir = os.path.join(OUT_ROOT, name, 'raw')
        split_dir = os.path.join(OUT_ROOT, name, 'splits')
        print('\n[{}] {}'.format(name, src['description']))
        for f in src['files']:
            dest = os.path.join(raw_dir, f['save_as'])
            try:
                _download(f['url'], dest)
            except Exception as e:
                print('  下载失败: {}'.format(e))
                sys.exit(1)

        first_file = os.path.join(raw_dir, src['files'][0]['save_as'])
        loader = LOADERS[src['loader']]
        df, tc, lc = loader(first_file)
        _split_save(df, tc, lc, split_dir)

    readme = os.path.join(DATA_NLP, 'README.txt')
    with open(readme, 'w', encoding='utf-8') as f:
        f.write(
            '本目录由 scripts/download_and_split_mental_health_datasets.py（英文）与手工/脚本整理的中文数据组成。\n\n'
            '划分比例（各 splits）：训练 70% / 验证 15% / 测试 15%，分层抽样 random_state=42。\n\n'
            '目录结构：\n'
            '  en/mental_health_4class/splits/*.csv  — 四分类（标签英文类别名）\n'
            '  en/reddit_depression_binary/splits/*.csv — 二分类 depression / non_depression\n'
            '  en/depression_posts_binary/splits/*.csv — 二分类 0/1\n'
            '  zh/oesd_keyword_binary/splits/*.csv — 中文关键词弱标注二分类（默认训练用）\n\n'
            'CSV 列：text, label\n\n'
            '英文数据来源：\n'
            '  - https://huggingface.co/datasets/nayma11/Mental-Health_Text-Classification_Dataset\n'
            '  - https://huggingface.co/datasets/hugginglearners/reddit-depression-cleaned\n'
            '  - https://huggingface.co/datasets/joangaes/depression\n'
            '中文 OESD：见 zh/oesd_keyword_binary/README.txt\n'
        )
    print('\n已写入说明: {}'.format(readme))
    print('全部完成。')


if __name__ == '__main__':
    main()
