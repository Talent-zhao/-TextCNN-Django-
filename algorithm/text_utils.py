# -*- coding: utf-8 -*-
import json
import logging
import os
import re

import jieba

# jieba 默认对「构建前缀词典 / 读 cache」等打 DEBUG，会污染训练流式日志与终端
jieba.setLogLevel(logging.WARNING)


def preprocess_pipeline(text):
    """
    与后台 PredictService 一致的预处理：去特殊字符 + jieba 分词 + 去停用词，空格拼接。
    用于 TF-IDF + sklearn / 统一预测流水线。
    """
    text = re.sub(r'[^\w\u4e00-\u9fff]+', '', text or '')
    stopword_path = 'stopwords.txt'
    stopword = []
    if os.path.exists(stopword_path):
        stopword = [i.strip() for i in open(stopword_path, 'r', encoding='utf-8').readlines()]
    seg_list = jieba.cut(text)
    return ' '.join([w for w in seg_list if w.strip() and w not in stopword])


def load_num2name(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def text_to_char_ids(text, vocab, max_len):
    tokens = [ch for ch in re.sub(r'\s+', '', text or '')]
    unk_id = vocab.get('<UNK>', 1)
    pad_id = vocab.get('<PAD>', 0)
    ids = [vocab.get(tok, unk_id) for tok in tokens[:max_len]]
    if len(ids) < max_len:
        ids.extend([pad_id] * (max_len - len(ids)))
    return ids
