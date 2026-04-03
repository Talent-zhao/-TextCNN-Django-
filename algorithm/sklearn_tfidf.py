# -*- coding: utf-8 -*-
import os

import joblib

from algorithm.text_utils import preprocess_pipeline, load_num2name


def predict_tfidf_sklearn(text, clf_path, vec_path, num2name_path, preprocess_fn=None):
    """
    使用已保存的 TF-IDF 向量器 + sklearn 分类器预测单条文本。

    返回 dict: enabled, available, label, prob, error
    （与 PredictService 各子模型返回结构一致，便于融合）
    """
    preprocess_fn = preprocess_fn or preprocess_pipeline
    if not clf_path or not vec_path:
        return {'enabled': True, 'available': False, 'label': None, 'prob': 0.0, 'error': 'path empty'}
    if not os.path.exists(clf_path) or not os.path.exists(vec_path):
        return {'enabled': True, 'available': False, 'label': None, 'prob': 0.0, 'error': 'model/vectorizer missing'}

    try:
        model = joblib.load(clf_path)
        vectorizer = joblib.load(vec_path)
        num2name = load_num2name(num2name_path or '')
        processed = preprocess_fn(text)
        x = vectorizer.transform([processed])
        pred_num = str(model.predict(x)[0])
        pred_name = num2name.get(pred_num, pred_num)
        prob = float(max(model.predict_proba(x)[0])) if hasattr(model, 'predict_proba') else 0.6
        return {'enabled': True, 'available': True, 'label': pred_name, 'prob': prob, 'error': ''}
    except Exception as e:
        return {'enabled': True, 'available': False, 'label': None, 'prob': 0.0, 'error': str(e)}
