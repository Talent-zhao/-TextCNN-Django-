# -*- coding: utf-8 -*-
"""
将 datasets/depression_nlp/{en|zh}/<数据集名>/splits/*.csv（列：text, label）
转换为后台「数据管理」页上传导入可用的 CSV（与 app/dataViews.import_rawtext_from_upload 字段一致）。
英文集在 en/ 下，中文默认划分在 zh/oesd_keyword_binary/splits/。

目标表头：content（必填）, external_id, name, time
- content  <- text
- external_id <- 唯一字符串（含数据集前缀与行号，便于追溯）
- name     <- 可选：把原始 label 写入，便于在「数据管理」列表里肉眼区分（后台不导入未知列）
- time     <- 留空（或可传固定占位，解析失败则 None）

用法（在项目根目录）:
  python scripts/splits_to_admin_import_csv.py \\
    --input datasets/depression_nlp/zh/oesd_keyword_binary/splits/train.csv \\
    --output data/import_mental_health_train.csv

  # 一次性生成 BUNDLES 中各 en/zh 数据集的 train（输出到 data/）
  python scripts/splits_to_admin_import_csv.py --all-train --out-dir data

依赖：pandas
"""
from __future__ import print_function

import argparse
import os
import re
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

DEFAULT_SPLITS_ROOT = os.path.join(ROOT, 'datasets', 'depression_nlp')

BUNDLES = [
    ('en/mental_health_4class', 'mh4c'),
    ('en/reddit_depression_binary', 'reddit_dep'),
    ('en/depression_posts_binary', 'posts_bin'),
    ('zh/oesd_keyword_binary', 'oesd_zh'),
]


def _safe_token(s):
    s = str(s).strip()
    s = re.sub(r'[^\w\-.]+', '_', s, flags=re.UNICODE)
    return s[:80] if s else 'x'


def convert_df(df, id_prefix):
    """text,label -> content,external_id,name,time"""
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError('输入 CSV 需包含列: text, label')
    out_rows = []
    for i, row in df.iterrows():
        text = (row['text'] if pd.notna(row['text']) else '') or ''
        text = str(text).strip()
        if not text:
            continue
        lab = row['label']
        lab_str = str(lab).strip() if pd.notna(lab) else ''
        ext = '{}_{}_{}'.format(id_prefix, i, _safe_token(lab_str))
        out_rows.append(
            {
                'content': text,
                'external_id': ext,
                'name': lab_str,
                'time': '',
            }
        )
    return pd.DataFrame(out_rows)


def run_one(input_path, output_path, id_prefix):
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    out = convert_df(df, id_prefix)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    out.to_csv(output_path, index=False, encoding='utf-8-sig')
    print('写入 {} 行 -> {}'.format(len(out), output_path))


def run_all_train(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for folder, prefix in BUNDLES:
        inp = os.path.join(DEFAULT_SPLITS_ROOT, folder, 'splits', 'train.csv')
        if not os.path.isfile(inp):
            print('跳过（不存在）: {}'.format(inp))
            continue
        outp = os.path.join(out_dir, 'import_{}_train.csv'.format(prefix))
        run_one(inp, outp, prefix)


def main():
    parser = argparse.ArgumentParser(description='splits CSV -> 后台 RawText 导入 CSV')
    parser.add_argument('--input', '-i', help='输入 train.csv / val.csv / test.csv')
    parser.add_argument('--output', '-o', help='输出 CSV 路径')
    parser.add_argument(
        '--id-prefix',
        default='import',
        help='external_id 前缀（默认 import）',
    )
    parser.add_argument(
        '--all-train',
        action='store_true',
        help='转换 depression_nlp 下三套数据集的 splits/train.csv 到 --out-dir',
    )
    parser.add_argument(
        '--out-dir',
        default=os.path.join(ROOT, 'data'),
        help='与 --all-train 配合，默认项目根下 data/',
    )
    args = parser.parse_args()

    if args.all_train:
        run_all_train(args.out_dir)
        return

    if not args.input or not args.output:
        parser.error('请指定 --input 与 --output，或使用 --all-train')

    if not os.path.isfile(args.input):
        print('文件不存在: {}'.format(args.input))
        sys.exit(1)
    run_one(args.input, args.output, args.id_prefix)


if __name__ == '__main__':
    main()
