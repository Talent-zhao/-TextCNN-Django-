# -*- coding: utf-8 -*-
"""
模型文件目录约定（相对项目根）：
  model/<算法>/   同算法训练产物放在同一子目录，便于管理。

  model/svm/     model_svm.pkl, tfidfVectorizer*.pkl
  model/knn/     …
  model/rf/, model/dt/, model/lr/
  model/textcnn/ textcnn.pt, textcnn_vocab.json
  model/textrcnn/ textrcnn.pt, textcnn_vocab.json（可与 textcnn 共用词表时请指向各自目录下文件）

  model/num2name.json          标签映射（各脚本统一写入根目录，便于融合默认引用）
  model/model_training_registry.json  训练注册表（仍在 model/ 根）
"""
import os


def sklearn_subdir(role):
    """统一使用正斜杠，便于注册表与跨平台。"""
    return 'model/{}'.format(role.lower())


def sklearn_model_rel(role):
    """如 model/svm/model_svm.pkl"""
    r = role.lower()
    return '{}/model_{}.pkl'.format(sklearn_subdir(r), r)


def sklearn_tfidf_primary_rel(role):
    """该算法目录下主向量器（与 train_single_sklearn 写出一致）"""
    r = role.lower()
    return '{}/tfidfVectorizer.pkl'.format(sklearn_subdir(r))


def sklearn_tfidf_role_rel(role):
    r = role.lower()
    return '{}/tfidfVectorizer_{}.pkl'.format(sklearn_subdir(r), r)


def legacy_sklearn_tfidf_role_rel(role):
    """旧版扁平路径，用于自动回退"""
    return 'model/tfidfVectorizer_{}.pkl'.format(role.lower())


def legacy_sklearn_model_rel(role):
    return 'model/model_{}.pkl'.format(role.lower())


NUM2NAME_REL = 'model/num2name.json'
REGISTRY_REL = 'model/model_training_registry.json'

TEXTCNN_DIR = 'model/textcnn'
TEXTRCNN_DIR = 'model/textrcnn'


def textcnn_weight_rel():
    return os.path.join(TEXTCNN_DIR, 'textcnn.pt').replace('\\', '/')


def textcnn_vocab_rel():
    return os.path.join(TEXTCNN_DIR, 'textcnn_vocab.json').replace('\\', '/')


def textrcnn_weight_rel():
    return os.path.join(TEXTRCNN_DIR, 'textrcnn.pt').replace('\\', '/')


def textrcnn_vocab_rel():
    """TextRCNN 词表默认与 textcnn 同文件名，放在 textrcnn 目录"""
    return os.path.join(TEXTRCNN_DIR, 'textcnn_vocab.json').replace('\\', '/')
