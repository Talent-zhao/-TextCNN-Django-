import json
import os

from django.conf import settings

from algorithm.model_paths import textcnn_vocab_rel, textcnn_weight_rel, textrcnn_vocab_rel, textrcnn_weight_rel
from app.models import ModelInfo
from app.services.predict_service import PredictService

try:
    import torch  # noqa: F401
except Exception:
    torch = None


def _project_rel_path(rel):
    if not rel:
        return rel
    if os.path.isabs(rel):
        return rel
    return os.path.join(settings.BASE_DIR, rel.replace('/', os.sep))


class ModelHealthService:
    LEVEL_OK = 'normal'
    LEVEL_WARN = 'warning'
    LEVEL_ERROR = 'error'

    @classmethod
    def _item(cls, level, check, message):
        return {'level': level, 'check': check, 'message': message}

    @classmethod
    def _model_meta(cls, model):
        """可 JSON 序列化的模型摘要（勿放入 ORM 对象）。"""
        if model is None:
            return None
        return {
            'id': model.id,
            'name': model.name,
            'version': model.version,
            'model_type': model.model_type,
            'model_type_display': model.get_model_type_display(),
        }

    @classmethod
    def _check_sklearn_pair(cls, model, cfg, role, label_zh, results):
        enable = bool(cfg.get('enable_{}'.format(role), False))
        if not enable:
            return
        clf = PredictService._resolve_clf_path(model, cfg, role)
        vec = PredictService._resolve_vec_path(model, cfg, role)

        clf_abs = _project_rel_path(clf)
        vec_abs = _project_rel_path(vec)
        if os.path.exists(clf_abs):
            results.append(cls._item(cls.LEVEL_OK, '{} 模型'.format(label_zh), '存在: {}'.format(clf)))
        else:
            results.append(cls._item(cls.LEVEL_ERROR, '{} 模型'.format(label_zh), '缺失: {}'.format(clf)))
        if os.path.exists(vec_abs):
            results.append(cls._item(cls.LEVEL_OK, '{} 向量器'.format(label_zh), '存在: {}'.format(vec)))
        else:
            results.append(cls._item(cls.LEVEL_ERROR, '{} 向量器'.format(label_zh), '缺失: {}'.format(vec)))

    @classmethod
    def _check_model(cls, model):
        results = []
        cfg = PredictService.get_runtime_config(model)

        results.append(cls._item(cls.LEVEL_OK, '模型基本信息', '{}/{}/{}'.format(model.name, model.version, model.model_type)))

        if model.config_json:
            try:
                json.loads(model.config_json)
                results.append(cls._item(cls.LEVEL_OK, 'config_json解析', '可解析'))
            except Exception as e:
                results.append(cls._item(cls.LEVEL_ERROR, 'config_json解析', '解析失败: {}'.format(e)))
        else:
            results.append(cls._item(cls.LEVEL_WARN, 'config_json解析', '为空，使用系统默认配置'))

        cls._check_sklearn_pair(model, cfg, 'svm', 'SVM', results)
        cls._check_sklearn_pair(model, cfg, 'knn', 'KNN', results)
        cls._check_sklearn_pair(model, cfg, 'rf', '随机森林', results)
        cls._check_sklearn_pair(model, cfg, 'dt', '决策树', results)
        cls._check_sklearn_pair(model, cfg, 'lr', '逻辑回归', results)

        textcnn_weight = str(cfg.get('textcnn_weight_path', textcnn_weight_rel()))
        textcnn_vocab = str(cfg.get('textcnn_vocab_path', textcnn_vocab_rel()))
        if cfg.get('enable_textcnn', True):
            torch_ok = torch is not None
            if torch_ok:
                results.append(cls._item(cls.LEVEL_OK, 'torch可用性', 'torch可用'))
            else:
                results.append(cls._item(cls.LEVEL_ERROR, 'torch可用性', 'torch不可用，将无法执行TextCNN推理'))
            tw = _project_rel_path(textcnn_weight)
            if os.path.exists(tw):
                results.append(cls._item(cls.LEVEL_OK, 'TextCNN权重', '存在: {}'.format(textcnn_weight)))
            else:
                results.append(cls._item(cls.LEVEL_ERROR, 'TextCNN权重', '缺失: {}'.format(textcnn_weight)))
            tv = _project_rel_path(textcnn_vocab)
            if os.path.exists(tv):
                results.append(cls._item(cls.LEVEL_OK, 'TextCNN词表', '存在: {}'.format(textcnn_vocab)))
            else:
                results.append(cls._item(cls.LEVEL_ERROR, 'TextCNN词表', '缺失: {}'.format(textcnn_vocab)))

        textrcnn_weight = str(cfg.get('textrcnn_weight_path', textrcnn_weight_rel()))
        textrcnn_vocab = str(cfg.get('textrcnn_vocab_path', cfg.get('textcnn_vocab_path', textrcnn_vocab_rel())))
        if cfg.get('enable_textrcnn', False):
            torch_ok = torch is not None
            if not torch_ok:
                results.append(cls._item(cls.LEVEL_ERROR, 'TextRCNN', 'torch不可用'))
            trw = _project_rel_path(textrcnn_weight)
            if os.path.exists(trw):
                results.append(cls._item(cls.LEVEL_OK, 'TextRCNN权重', '存在: {}'.format(textrcnn_weight)))
            else:
                results.append(cls._item(cls.LEVEL_ERROR, 'TextRCNN权重', '缺失: {}'.format(textrcnn_weight)))
            trv = _project_rel_path(textrcnn_vocab)
            if os.path.exists(trv):
                results.append(cls._item(cls.LEVEL_OK, 'TextRCNN词表', '存在: {}'.format(textrcnn_vocab)))
            else:
                results.append(cls._item(cls.LEVEL_ERROR, 'TextRCNN词表', '缺失: {}'.format(textrcnn_vocab)))
        has_error = any(i['level'] == cls.LEVEL_ERROR for i in results)
        has_warn = any(i['level'] == cls.LEVEL_WARN for i in results)
        summary = cls.LEVEL_ERROR if has_error else (cls.LEVEL_WARN if has_warn else cls.LEVEL_OK)
        return {'model': cls._model_meta(model), 'summary': summary, 'items': results}

    @classmethod
    def check_active_model(cls):
        active = ModelInfo.objects.filter(is_active=True, status='ready').order_by('-created_at').first()
        if not active:
            return {
                'model': None,
                'summary': cls.LEVEL_ERROR,
                'items': [cls._item(cls.LEVEL_ERROR, '当前启用模型', '不存在可用的启用模型')],
            }
        return cls._check_model(active)

    @classmethod
    def check_all_models(cls):
        rows = ModelInfo.objects.all().order_by('-created_at')
        if not rows.exists():
            return [{
                'model': None,
                'summary': cls.LEVEL_ERROR,
                'items': [cls._item(cls.LEVEL_ERROR, '模型列表', '系统中暂无模型配置')],
            }]
        return [cls._check_model(m) for m in rows]
