# -*- coding: utf-8 -*-
"""
下载 HuggingFace 上 ourafla/Mental-Health_Text-Classification_Dataset 的两份 CSV
并支持断点续传，避免大文件下载失败后需要重头来过。

把文件落到：
datasets/depression_nlp/zh/mental_health_4class_ourafla_zh/raw/
"""

from __future__ import annotations

import os
import sys
import time
from typing import Tuple

import requests


URLS: Tuple[Tuple[str, str], ...] = (
    (
        "mental_heath_unbanlanced.csv",
        "https://huggingface.co/datasets/ourafla/Mental-Health_Text-Classification_Dataset/resolve/main/mental_heath_unbanlanced.csv",
    ),
    (
        "mental_health_combined_test.csv",
        "https://huggingface.co/datasets/ourafla/Mental-Health_Text-Classification_Dataset/resolve/main/mental_health_combined_test.csv",
    ),
)


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _download_one(url: str, out_path: str, retry: int = 3) -> None:
    sess = requests.Session()
    sess.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MentalHealthDatasetDownloader/1.0",
        }
    )

    for attempt in range(1, retry + 1):
        existing = os.path.getsize(out_path) if os.path.isfile(out_path) else 0
        headers = {}
        mode = "wb"
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
            mode = "ab"

        try:
            print(f"[download] attempt {attempt}/{retry}: {os.path.basename(out_path)} existing={existing}", flush=True)
            with sess.get(url, stream=True, headers=headers, timeout=(30, 600)) as r:
                # 如果断点续传失败（服务器不支持 range），则回退为全量覆盖。
                if existing > 0 and r.status_code not in (200, 206):
                    print(f"[download] server returned {r.status_code}, restart full download.", flush=True)
                    existing = 0
                    mode = "wb"
                    headers = {}
                    r.close()
                    continue

                if r.status_code == 416:
                    # already complete
                    print("[download] already complete (416).", flush=True)
                    return

                r.raise_for_status()

                total = None
                cl = r.headers.get("Content-Length")
                if cl and cl.isdigit():
                    total = int(cl) + existing

                downloaded = existing
                chunk_size = 1024 * 1024  # 1MB
                start = time.time()
                with open(out_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded * 100.0 / total
                            elapsed = time.time() - start
                            speed = downloaded / max(elapsed, 1e-6) / (1024 * 1024)
                            print(f"[download] {pct:6.2f}% {speed:8.2f} MB/s", flush=True)
                print(f"[download] done: {out_path} bytes={downloaded}", flush=True)
                return
        except Exception as e:
            print(f"[download] failed attempt {attempt}: {repr(e)}", flush=True)
            # small backoff
            time.sleep(3 * attempt)
            continue

    raise RuntimeError(f"Download failed after {retry} attempts: {out_path}")


def main() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    raw_dir = os.path.join(project_root, "datasets", "depression_nlp", "zh", "mental_health_4class_ourafla_zh", "raw")
    _ensure_dir(raw_dir)

    for fname, url in URLS:
        out_path = os.path.join(raw_dir, fname)
        _download_one(url, out_path)


if __name__ == "__main__":
    main()

