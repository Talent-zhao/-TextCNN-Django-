# -*- coding: utf-8 -*-
"""从 Hugging Face 下载二分类/四分类抑郁相关文本模型到本目录。"""
import os

# 国内可设: set HF_ENDPOINT=https://hf-mirror.com
try:
    from huggingface_hub import snapshot_download
except ImportError:
    raise SystemExit("请先安装: pip install huggingface_hub")

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = (
    ("TRT1000/depression-detection-model", "binary_depression_distilbert"),
    ("ourafla/mental-health-bert-finetuned", "four_class_mental_health_bert"),
)


def main():
    for repo_id, sub in MODELS:
        dest = os.path.join(HERE, sub)
        print("Downloading", repo_id, "->", dest)
        snapshot_download(repo_id=repo_id, local_dir=dest)
    print("完成。")


if __name__ == "__main__":
    main()
