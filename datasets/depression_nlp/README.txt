datasets/depression_nlp 目录说明
================================

en/   英文心理健康/抑郁相关数据集（可由 scripts/download_and_split_mental_health_datasets.py 下载并划分）
  mental_health_4class/   四分类，标签为英文类别名
  reddit_depression_binary/
  depression_posts_binary/

zh/   中文数据（默认在线训练页、脚本默认 CSV）
  oesd_keyword_binary/    OESD-GG-zh_cn-1，弱标注二分类；见该目录 README.txt

划分比例（各数据集 splits/）：训练 70% / 验证 15% / 测试 15%，分层抽样 random_state=42。

CSV 列统一为：text, label

目录整理、英文迁入 en/、中文划分迁入 zh/ 可执行：
  python scripts/reorganize_depression_nlp_en_zh.py

后台「数据导入」可由 splits 转 CSV：
  python scripts/splits_to_admin_import_csv.py --all-train --out-dir data
