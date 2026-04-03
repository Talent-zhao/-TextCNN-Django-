# -*- coding: utf-8 -*-
"""
算法与预测相关实现（与 Django ORM 解耦，供 app.services.predict_service 等调用）。

包含：
- 文本预处理
- TF-IDF + sklearn 分类器（SVM / KNN / 随机森林 / 决策树 等）
- PyTorch TextCNN / TextRCNN
- 规则打分与分数融合
"""

from algorithm.text_utils import preprocess_pipeline, load_num2name, text_to_char_ids
from algorithm.sklearn_tfidf import predict_tfidf_sklearn
from algorithm.torch_nlp import predict_textcnn, predict_textrcnn
from algorithm.rules import predict_rules, HIGH_RISK_RULE_WORDS, NEGATIVE_WORDS
from algorithm.fusion import fuse_weighted_scores

__all__ = [
    'preprocess_pipeline',
    'load_num2name',
    'text_to_char_ids',
    'predict_tfidf_sklearn',
    'predict_textcnn',
    'predict_textrcnn',
    'predict_rules',
    'HIGH_RISK_RULE_WORDS',
    'NEGATIVE_WORDS',
    'fuse_weighted_scores',
]
