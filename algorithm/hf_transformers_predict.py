# -*- coding: utf-8 -*-
import os

from django.conf import settings

_CACHE = {}


def predict_hf_sequence_classifier(local_dir_rel, text, max_length=512):
    """
    使用本地下载的 HuggingFace Transformers 序列分类目录预测单条文本。
    local_dir_rel: 相对项目根目录，如 Best_Modle/binary_depression_distilbert
    返回: dict label(展示名), prob, pred_id, error, raw_label(模型原始 id2label)
    """
    out = {'label': None, 'prob': 0.0, 'pred_id': None, 'error': '', 'raw_label': None}
    if not local_dir_rel or not text:
        out['error'] = 'empty path or text'
        return out
    path = os.path.join(settings.BASE_DIR, local_dir_rel.replace('/', os.sep))
    if not os.path.isdir(path):
        out['error'] = '模型目录不存在: {}'.format(path)
        return out
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
    except ImportError as e:
        out['error'] = '缺少依赖，请安装: pip install transformers torch safetensors（{}）'.format(e)
        return out

    if path not in _CACHE:
        tok = AutoTokenizer.from_pretrained(path)
        mdl = AutoModelForSequenceClassification.from_pretrained(path)
        mdl.eval()
        _CACHE[path] = (tok, mdl)
    tok, mdl = _CACHE[path]

    enc = tok(text, return_tensors='pt', truncation=True, max_length=max_length)
    with torch.no_grad():
        logits = mdl(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred = int(torch.argmax(probs).item())
        prob = float(probs[pred].item())

    raw = None
    id2label = getattr(mdl.config, 'id2label', None)
    if id2label is not None:
        raw = id2label.get(str(pred), id2label.get(pred))

    out['pred_id'] = pred
    out['prob'] = prob
    out['raw_label'] = raw
    out['label'] = raw if raw and not str(raw).startswith('LABEL_') else str(pred)
    return out


def map_binary_depression_label(pred_id):
    """0=非抑郁 1=抑郁 -> 与库中 LabelDefinition 名称一致."""
    return 'HF_NotDepressed' if pred_id == 0 else 'HF_Depressed'


def map_four_class_mental_health_label(pred_id):
    """与 ourafla 模型卡一致的类别顺序."""
    names = {
        0: 'HF_Anxiety',
        1: 'HF_Depression',
        2: 'HF_Normal',
        3: 'HF_Suicidal',
    }
    return names.get(pred_id, 'HF_Normal')
