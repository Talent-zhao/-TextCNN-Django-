# -*- coding: utf-8 -*-
"""
一次性：将 datasets/depression_nlp 下英文数据迁入 en/，中文 OESCD 迁入 zh/ 并划分 train/val/test。

用法（项目ROOT）:
  python scripts/reorganize_depression_nlp_en_zh.py

说明：
  - 英文：mental_health_4class、reddit_depression_binary、depression_posts_binary -> en/<name>/
  - 中文：oesd_zh -> zh/oesd_keyword_binary/（parquet 入 raw/；keyword 全量 CSV 入 raw/；splits 7:1.5:1.5）
  - 可选更新 model_training_registry.json 内 last_csv 路径前缀
"""
from __future__ import print_function

import json
import os
import shutil
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
BASE = os.path.join(ROOT, 'datasets', 'depression_nlp')
EN = os.path.join(BASE, 'en')
ZH = os.path.join(BASE, 'zh')

ENGLISH_NAMES = ('mental_health_4class', 'reddit_depression_binary', 'depression_posts_binary')


def _split_save(df, split_dir, random_state=42):
    text_col, label_col = 'text', 'label'
    df = df[[text_col, label_col]].copy()
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
    print('  zh splits: train={} val={} test={}'.format(len(X_train), len(X_val), len(X_test)))


def _move_if_exists(src, dst_parent):
    if not os.path.isdir(src):
        return
    name_want = os.path.basename(os.path.normpath(src))
    dst = os.path.join(dst_parent, name_want)
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    if os.path.exists(dst):
        print('  已存在，跳过移动: {}'.format(dst))
        return
    os.makedirs(dst_parent, exist_ok=True)
    shutil.move(src, dst)
    print('  已移动: {} -> {}'.format(src, dst))


def _update_registry_paths():
    reg_path = os.path.join(ROOT, 'model', 'model_training_registry.json')
    if not os.path.isfile(reg_path):
        return
    try:
        with open(reg_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print('  注册表跳过（读取失败）:', e)
        return
    if not isinstance(data, dict):
        return
    repl = [
        ('datasets/depression_nlp/mental_health_4class/', 'datasets/depression_nlp/en/mental_health_4class/'),
        ('datasets/depression_nlp/reddit_depression_binary/', 'datasets/depression_nlp/en/reddit_depression_binary/'),
        ('datasets/depression_nlp/depression_posts_binary/', 'datasets/depression_nlp/en/depression_posts_binary/'),
        ('datasets/depression_nlp/oesd_zh/', 'datasets/depression_nlp/zh/oesd_keyword_binary/'),
    ]

    def fix_text(s):
        if not s or not isinstance(s, str):
            return s
        out = s
        for a, b in repl:
            out = out.replace(a, b)
        return out

    changed = False
    for _k, row in data.items():
        if not isinstance(row, dict):
            continue
        lc = row.get('last_csv')
        if lc:
            n = fix_text(lc)
            if n != lc:
                row['last_csv'] = n
                changed = True
    if changed:
        with open(reg_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print('  已更新 model_training_registry.json 内 last_csv 前缀')


def main():
    os.makedirs(EN, exist_ok=True)
    os.makedirs(ZH, exist_ok=True)

    print('[1] 英文数据集 -> en/')
    for name in ENGLISH_NAMES:
        _move_if_exists(os.path.join(BASE, name), EN)

    print('[2] 中文 OESD -> zh/oesd_keyword_binary/')
    src_oesd = os.path.join(BASE, 'oesd_zh')
    zdst = os.path.join(ZH, 'oesd_keyword_binary')
    raw_d = os.path.join(zdst, 'raw')
    spl_d = os.path.join(zdst, 'splits')
    os.makedirs(raw_d, exist_ok=True)

    parq_src = os.path.join(src_oesd, 'train-0000.parquet')
    if os.path.isfile(parq_src):
        parq_dst = os.path.join(raw_d, 'train-0000.parquet')
        if not os.path.isfile(parq_dst):
            shutil.move(parq_src, parq_dst)
            print('  parquet ->', parq_dst)
        else:
            print('  parquet 已在目标，跳过')

    keyword_src = os.path.join(src_oesd, 'splits', 'train_keyword_binary.csv')
    if os.path.isfile(keyword_src):
        all_dst = os.path.join(raw_d, 'all_keyword_binary.csv')
        shutil.copy2(keyword_src, all_dst)
        print('  全量 keyword CSV 复制 ->', all_dst)
        df = pd.read_csv(all_dst, encoding='utf-8-sig')
        _split_save(df, spl_d)
    else:
        # 若已在 zh 下只有 parquet，可再生成 keyword（需与训练侧一致时用户可单独跑）
        all_in_zh = os.path.join(raw_d, 'all_keyword_binary.csv')
        if os.path.isfile(all_in_zh):
            df = pd.read_csv(all_in_zh, encoding='utf-8-sig')
            _split_save(df, spl_d)

    for fn in ('README.txt',):
        p = os.path.join(src_oesd, fn)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(zdst, fn))

    # 删除已搬空目录 oesd_zh
    if os.path.isdir(src_oesd):
        try:
            left = os.listdir(src_oesd)
            if not left:
                os.rmdir(src_oesd)
                print('  已删除空目录 oesd_zh')
            else:
                print('  提醒: oesd_zh 下仍有文件，请手动检查:', left)
        except OSError:
            pass

    print('[3] 注册表路径')
    _update_registry_paths()
    print('完成。')


if __name__ == '__main__':
    main()
