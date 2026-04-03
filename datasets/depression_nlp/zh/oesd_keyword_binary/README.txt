OESD-GG-zh_cn-1（中文）— 关键词弱标注二分类

来源（许可证以官方为准）：
  https://huggingface.co/datasets/depression-icu/OESD-GG-zh_cn-1

本目录：
  raw/train-0000.parquet     原始 Parquet（列 User, Assisstant）
  raw/all_keyword_binary.csv 全量弱标注（text, label）
  splits/train.csv|val.csv|test.csv  由 all_keyword_binary 去重后 7:1.5:1.5 划分

label 含义（非临床诊断）：
  depression_related — 用户文本命中少量抑郁/轻生相关词
  other — 其余

说明：User 列在去重后约四千余条唯一文本（原数据存在大量重复用户话术），划分基于去重后样本。
