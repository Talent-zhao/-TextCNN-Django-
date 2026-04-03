# -*- coding: utf-8 -*-
"""
将 HuggingFace 的 4-class 心理健康数据集导出成“中文子集”，并落到项目 datasets 目录供训练使用。

由于本环境对 HuggingFace 大文件抓取可能超时：
- 你需要先手动下载 CSV 到 raw 目录（脚本只负责过滤/导出）。

数据源（HuggingFace）：
ourafla/Mental-Health_Text-Classification_Dataset
文件名（目录内需与下面一致）：
- raw/mental_heath_unbanlanced.csv
- raw/mental_health_combined_test.csv

输出（按项目约定格式）：
datasets/depression_nlp/zh/mental_health_4class_ourafla_zh/splits/train.csv
datasets/depression_nlp/zh/mental_health_4class_ourafla_zh/splits/test.csv
导出列：text,label

标签映射：
Suicidal -> HF_Suicidal
Depression -> HF_Depression
Anxiety -> HF_Anxiety
Normal -> HF_Normal
"""

from __future__ import annotations

import os
import re
import sys

import pandas as pd


LABEL_MAP = {
    "Suicidal": "HF_Suicidal",
    "Depression": "HF_Depression",
    "Anxiety": "HF_Anxiety",
    "Normal": "HF_Normal",
}


def _contains_chinese(text: str) -> bool:
    # 至少出现一个中文汉字
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "text" not in df.columns or "status" not in df.columns:
        # 数据集有的版本叫 label/status，这里兜底
        if "label" in df.columns:
            df = df.rename(columns={"label": "status"})
    if "text" not in df.columns or "status" not in df.columns:
        raise ValueError(f"CSV 缺少列：需要 text + status（或 label），实际列：{list(df.columns)}")
    df["text"] = df["text"].astype(str)
    df["status"] = df["status"].astype(str)
    return df


def prepare(
    raw_dir: str,
    out_dir: str,
    min_chars_for_keep: int = 1,
) -> None:
    raw_train = os.path.join(raw_dir, "mental_heath_unbanlanced.csv")
    raw_test = os.path.join(raw_dir, "mental_health_combined_test.csv")

    train_df = _read_csv(raw_train)
    test_df = _read_csv(raw_test)

    # 映射为项目 HF 标签名（供 LabelDefinition 查找）
    def map_status(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df[df["status"].isin(LABEL_MAP.keys())]
        df["label"] = df["status"].map(LABEL_MAP)
        return df[["text", "label"]]

    train_df = map_status(train_df)
    test_df = map_status(test_df)

    # 中文过滤：至少含一个汉字
    train_df = train_df[train_df["text"].str.len() >= min_chars_for_keep]
    train_df = train_df[train_df["text"].apply(_contains_chinese)]
    test_df = test_df[test_df["text"].str.len() >= min_chars_for_keep]
    test_df = test_df[test_df["text"].apply(_contains_chinese)]

    # 输出
    splits_dir = os.path.join(out_dir, "splits")
    _ensure_dir(splits_dir)
    train_out = os.path.join(splits_dir, "train.csv")
    test_out = os.path.join(splits_dir, "test.csv")

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(train_out, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_out, index=False, encoding="utf-8-sig")

    print("[zh_prepare] done.")
    print("[zh_prepare] train rows =", len(train_df), "->", train_out)
    print("[zh_prepare] test rows  =", len(test_df), "->", test_out)


if __name__ == "__main__":
    # 固定落地目录：datasets/depression_nlp/zh/mental_health_4class_ourafla_zh
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    raw_dir = os.path.join(PROJECT_ROOT, "datasets", "depression_nlp", "zh", "mental_health_4class_ourafla_zh", "raw")
    out_dir = os.path.join(PROJECT_ROOT, "datasets", "depression_nlp", "zh", "mental_health_4class_ourafla_zh")

    min_chars = 1
    # 用法示例：
    #   python scripts/prepare_ourafla_4class_zh_subset.py --raw-dir D:\download --min-chars 2
    if len(sys.argv) >= 2:
        i = 1
        while i < len(sys.argv):
            if sys.argv[i] == "--raw-dir" and i + 1 < len(sys.argv):
                raw_dir = sys.argv[i + 1]
                i += 2
                continue
            if sys.argv[i] == "--min-chars" and i + 1 < len(sys.argv):
                min_chars = int(sys.argv[i + 1])
                i += 2
                continue
            i += 1

    prepare(raw_dir=raw_dir, out_dir=out_dir, min_chars_for_keep=min_chars)

