import json
import logging
import numbers
import os
import traceback

from django.conf import settings

from app.models import LabelDefinition, ModelInfo, PredictionResult, RawText, SystemConfig

from algorithm.fusion import finite_float, fuse_weighted_scores
from algorithm.model_paths import (
    NUM2NAME_REL,
    legacy_sklearn_model_rel,
    legacy_sklearn_tfidf_role_rel,
    sklearn_model_rel,
    sklearn_tfidf_primary_rel,
    sklearn_tfidf_role_rel,
    textcnn_vocab_rel,
    textcnn_weight_rel,
    textrcnn_vocab_rel,
    textrcnn_weight_rel,
)
from algorithm.hf_transformers_predict import (
    map_binary_depression_label,
    map_four_class_mental_health_label,
    predict_hf_sequence_classifier,
)
from algorithm.rules import predict_rules
from algorithm.sklearn_tfidf import predict_tfidf_sklearn
from algorithm.text_utils import preprocess_pipeline

HF_MODEL_TYPES = ('hf_depr_bin', 'hf_mh_4cls')
TORCH_MODEL_TYPES = HF_MODEL_TYPES + ('textcnn', 'textrcnn')

logger = logging.getLogger(__name__)


def _json_clean_for_detail(obj):
    """保证 prediction_detail 可 json.dumps（去掉 numpy 标量等）。"""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _json_clean_for_detail(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean_for_detail(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, numbers.Integral):
        return int(obj)
    if isinstance(obj, numbers.Real):
        return float(obj)
    return str(obj)


SKLEARN_ROLES = ('svm', 'knn', 'rf', 'dt', 'lr')
ROLE_DEFAULT_PATH = {
    'svm': ('svm_model_path', sklearn_model_rel('svm')),
    'knn': ('knn_model_path', sklearn_model_rel('knn')),
    'rf': ('rf_model_path', sklearn_model_rel('rf')),
    'dt': ('dt_model_path', sklearn_model_rel('dt')),
    'lr': ('lr_model_path', sklearn_model_rel('lr')),
}


class PredictService:
    """
    统一预测执行器：调用 algorithm 包中的各子模型，融合后写入 PredictionResult。
    """

    DEFAULT_RUNTIME_CONFIG = {
        'enable_svm': True,
        'enable_knn': False,
        'enable_rf': False,
        'enable_dt': False,
        'enable_lr': False,
        'enable_textcnn': True,
        'enable_textrcnn': False,
        'enable_rules': True,
        'weight_svm': 0.35,
        'weight_knn': 0.1,
        'weight_rf': 0.1,
        'weight_dt': 0.1,
        'weight_lr': 0.1,
        'weight_textcnn': 0.15,
        'weight_textrcnn': 0.1,
        'weight_rules': 0.1,
        'threshold_high_risk': 0.85,
        'threshold_alert': 0.65,
        'threshold_rule_alert': 0.5,
        'tfidf_vectorizer_path': sklearn_tfidf_primary_rel('svm'),
        'svm_model_path': sklearn_model_rel('svm'),
        'knn_model_path': sklearn_model_rel('knn'),
        'rf_model_path': sklearn_model_rel('rf'),
        'dt_model_path': sklearn_model_rel('dt'),
        'lr_model_path': sklearn_model_rel('lr'),
        'textcnn_vocab_path': textcnn_vocab_rel(),
        'textcnn_weight_path': textcnn_weight_rel(),
        'textrcnn_vocab_path': textrcnn_vocab_rel(),
        'textrcnn_weight_path': textrcnn_weight_rel(),
        'num2name_path': NUM2NAME_REL,
        'textcnn_max_len': 128,
        'textcnn_embedding_dim': 128,
        'textcnn_num_classes': 4,
        'textcnn_kernel_sizes': '3,4,5',
        'textcnn_num_channels': 100,
        'textcnn_dropout': 0.5,
        'textrcnn_max_len': 128,
        'textrcnn_embedding_dim': 128,
        'textrcnn_hidden_dim': 256,
        'textrcnn_num_classes': 4,
        'textrcnn_dropout': 0.5,
    }

    @classmethod
    def _get_cfg_value(cls, key, default):
        row = SystemConfig.objects.filter(key=key).first()
        if not row:
            return default
        raw = row.value
        if isinstance(default, bool):
            return str(raw).lower() in ('1', 'true', 'yes', 'on')
        if isinstance(default, int):
            try:
                return int(raw)
            except Exception:
                return default
        if isinstance(default, float):
            try:
                return float(raw)
            except Exception:
                return default
        return raw

    @classmethod
    def ensure_default_runtime_config(cls):
        for k, v in cls.DEFAULT_RUNTIME_CONFIG.items():
            SystemConfig.objects.get_or_create(
                key=k,
                defaults={
                    'value': str(v),
                    'value_type': type(v).__name__,
                    'description': '预测融合运行配置',
                },
            )

    @classmethod
    def get_runtime_config(cls, model_info=None):
        cls.ensure_default_runtime_config()
        cfg = {}
        for k, default in cls.DEFAULT_RUNTIME_CONFIG.items():
            cfg[k] = cls._get_cfg_value(k, default)

        if model_info:
            mt = model_info.model_type
            if mt == 'svm':
                cfg.update({f'enable_{r}': (r == 'svm') for r in SKLEARN_ROLES})
                cfg['enable_textcnn'] = False
                cfg['enable_textrcnn'] = False
                cfg['enable_rules'] = True
            elif mt == 'knn':
                cfg.update({f'enable_{r}': (r == 'knn') for r in SKLEARN_ROLES})
                cfg['enable_textcnn'] = False
                cfg['enable_textrcnn'] = False
                cfg['enable_rules'] = True
            elif mt == 'rf':
                cfg.update({f'enable_{r}': (r == 'rf') for r in SKLEARN_ROLES})
                cfg['enable_textcnn'] = False
                cfg['enable_textrcnn'] = False
                cfg['enable_rules'] = True
            elif mt == 'dt':
                cfg.update({f'enable_{r}': (r == 'dt') for r in SKLEARN_ROLES})
                cfg['enable_textcnn'] = False
                cfg['enable_textrcnn'] = False
                cfg['enable_rules'] = True
            elif mt == 'lr':
                cfg.update({f'enable_{r}': (r == 'lr') for r in SKLEARN_ROLES})
                cfg['enable_textcnn'] = False
                cfg['enable_textrcnn'] = False
                cfg['enable_rules'] = True
            elif mt == 'textcnn':
                cfg.update({f'enable_{r}': False for r in SKLEARN_ROLES})
                cfg['enable_textcnn'] = True
                cfg['enable_textrcnn'] = False
                cfg['enable_rules'] = True
            elif mt == 'textrcnn':
                cfg.update({f'enable_{r}': False for r in SKLEARN_ROLES})
                cfg['enable_textcnn'] = False
                cfg['enable_textrcnn'] = True
                cfg['enable_rules'] = True
            elif mt == 'rule':
                cfg.update({f'enable_{r}': False for r in SKLEARN_ROLES})
                cfg['enable_textcnn'] = False
                cfg['enable_textrcnn'] = False
                cfg['enable_rules'] = True
            elif mt == 'fusion':
                # 融合策略登记：子模型开关与权重以 SystemConfig 为准，不在此强行打开全部算法
                pass
            elif mt in HF_MODEL_TYPES:
                cfg.update({f'enable_{r}': False for r in SKLEARN_ROLES})
                cfg['enable_textcnn'] = False
                cfg['enable_textrcnn'] = False
                cfg['enable_rules'] = True

        if model_info and model_info.config_json:
            try:
                cfg.update(json.loads(model_info.config_json))
            except Exception as e:
                logger.warning(
                    'model_info.config_json 解析失败，已忽略（id=%s name=%s）: %s',
                    getattr(model_info, 'id', None),
                    getattr(model_info, 'name', None),
                    e,
                )
        # 当父进程显式要求安全模式时，禁用所有需要 torch 的子模型，避免 Windows 原生崩溃。
        if os.environ.get('DISABLE_TORCH_MODELS') == '1':
            cfg['enable_textcnn'] = False
            cfg['enable_textrcnn'] = False
        return cfg

    @classmethod
    def load_active_model(cls):
        active = ModelInfo.objects.filter(is_active=True, status='ready').order_by('-created_at').first()
        if active:
            return active
        default_model, _ = ModelInfo.objects.get_or_create(
            name='svm_baseline',
            version='v1',
            defaults={
                'model_type': 'svm',
                'status': 'ready',
                'is_active': True,
                'listed_for_users': True,
                'file_path': sklearn_model_rel('svm'),
                'vectorizer_path': sklearn_tfidf_primary_rel('svm'),
                'description': '默认SVM基线模型',
                'config_json': '{}',
                'metrics_json': '{}',
            },
        )
        if not default_model.is_active:
            default_model.is_active = True
            default_model.save(update_fields=['is_active'])
        return default_model

    @classmethod
    def load_non_torch_model(cls):
        """
        返回一个不依赖 torch 的可用模型（优先 ready + active），用于 native 崩溃后的安全降级。
        """
        row = (
            ModelInfo.objects.filter(status='ready')
            .exclude(model_type__in=TORCH_MODEL_TYPES)
            .order_by('-is_active', '-created_at')
            .first()
        )
        if row:
            return row
        # 兜底返回默认 SVM，但不强制改动当前激活模型。
        default_model, _ = ModelInfo.objects.get_or_create(
            name='svm_baseline',
            version='v1',
            defaults={
                'model_type': 'svm',
                'status': 'ready',
                'is_active': True,
                'listed_for_users': True,
                'file_path': sklearn_model_rel('svm'),
                'vectorizer_path': sklearn_tfidf_primary_rel('svm'),
                'description': '默认SVM基线模型',
                'config_json': '{}',
                'metrics_json': '{}',
            },
        )
        return default_model

    @classmethod
    def _resolve_model_for_run(cls, model_info_id=None):
        """
        解析本次预测应使用的模型：
        - 指定 model_info_id 时优先使用该模型（要求 ready）
        - 未指定时使用当前 active
        - 环境变量要求禁用 torch/HF 时自动回退
        """
        model_info = None
        if model_info_id is not None:
            model_info = ModelInfo.objects.filter(id=model_info_id, status='ready').first()
            if not model_info:
                raise ValueError('所选模型不存在或不可用: {}'.format(model_info_id))
        if model_info is None:
            model_info = cls.load_active_model()
        if os.environ.get('DISABLE_HF_MODELS') == '1' and model_info and model_info.model_type in HF_MODEL_TYPES:
            model_info = cls.load_non_torch_model()
        if os.environ.get('DISABLE_TORCH_MODELS') == '1' and model_info and model_info.model_type in TORCH_MODEL_TYPES:
            model_info = cls.load_non_torch_model()
        return model_info

    @classmethod
    def _resolve_vec_path(cls, model_info, cfg, role=None):
        """
        融合或多算法并存时，每个分类器应对应其训练时保存的 TF-IDF（tfidfVectorizer_{role}.pkl），
        否则会与共用 tfidfVectorizer.pkl 但不同训练快照的 clf 不匹配。
        """
        if model_info and model_info.model_type in SKLEARN_ROLES and model_info.vectorizer_path:
            return model_info.vectorizer_path
        cfg_key = 'tfidf_vectorizer_path_{}'.format(role) if role else None
        if role and cfg_key and cfg.get(cfg_key):
            return cfg[cfg_key]
        if role:
            for rel in (
                sklearn_tfidf_role_rel(role),
                sklearn_tfidf_primary_rel(role),
                legacy_sklearn_tfidf_role_rel(role),
            ):
                full = os.path.join(settings.BASE_DIR, rel.replace('/', os.sep))
                if os.path.isfile(full):
                    return rel
        return (
            cfg.get('tfidf_vectorizer_path')
            or (model_info.vectorizer_path if model_info else None)
            or sklearn_tfidf_primary_rel('svm')
        )

    @classmethod
    def _resolve_clf_path(cls, model_info, cfg, role):
        cfg_key, default_path = ROLE_DEFAULT_PATH[role]
        if model_info and model_info.model_type == role:
            primary = (model_info.file_path or default_path).strip()
        else:
            primary = (cfg.get(cfg_key) or default_path).strip()
        for rel in (primary, legacy_sklearn_model_rel(role), default_path):
            if not rel:
                continue
            full = os.path.join(settings.BASE_DIR, rel.replace('/', os.sep))
            if os.path.isfile(full):
                return rel
        return primary or default_path

    @classmethod
    def _predict_sklearn_role(cls, text, model_info, cfg, role):
        enable_key = 'enable_{}'.format(role)
        if not cfg.get(enable_key, False):
            return {'enabled': False, 'available': False, 'label': None, 'prob': 0.0, 'error': '{} disabled'.format(role)}
        clf = cls._resolve_clf_path(model_info, cfg, role)
        vec = cls._resolve_vec_path(model_info, cfg, role)
        num = cfg.get('num2name_path', NUM2NAME_REL)
        ret = predict_tfidf_sklearn(text, clf, vec, num, preprocess_pipeline)
        return ret

    @classmethod
    def predict_with_svm(cls, text, model_info, cfg):
        return cls._predict_sklearn_role(text, model_info, cfg, 'svm')

    @classmethod
    def predict_with_knn(cls, text, model_info, cfg):
        return cls._predict_sklearn_role(text, model_info, cfg, 'knn')

    @classmethod
    def predict_with_rf(cls, text, model_info, cfg):
        return cls._predict_sklearn_role(text, model_info, cfg, 'rf')

    @classmethod
    def predict_with_dt(cls, text, model_info, cfg):
        return cls._predict_sklearn_role(text, model_info, cfg, 'dt')

    @classmethod
    def predict_with_lr(cls, text, model_info, cfg):
        return cls._predict_sklearn_role(text, model_info, cfg, 'lr')

    @classmethod
    def _resolve_existing_rel(cls, *rels):
        """按顺序返回第一个在磁盘存在的相对路径（相对 BASE_DIR），均无则返回第一个非空项。"""
        first = None
        for rel in rels:
            if not rel:
                continue
            if first is None:
                first = rel
            full = os.path.join(settings.BASE_DIR, rel.replace('/', os.sep))
            if os.path.isfile(full):
                return rel
        return first

    @classmethod
    def predict_with_textcnn(cls, text, model_info, cfg):
        if not cfg.get('enable_textcnn', True):
            return {'enabled': False, 'available': False, 'label': None, 'prob': 0.0, 'error': 'textcnn disabled'}
        # 延迟导入，避免在仅使用传统模型时触发 torch 导入崩溃。
        from algorithm.torch_nlp import predict_textcnn

        vocab = cfg.get('textcnn_vocab_path') or (model_info.vectorizer_path if model_info else None) or textcnn_vocab_rel()
        weight = cfg.get('textcnn_weight_path') or (
            model_info.file_path if model_info and model_info.model_type == 'textcnn' else None
        ) or textcnn_weight_rel()
        vocab = cls._resolve_existing_rel(vocab, 'model/textcnn_vocab.json')
        weight = cls._resolve_existing_rel(weight, 'model/textcnn.pt')
        return predict_textcnn(
            text,
            weight,
            vocab,
            cfg.get('num2name_path', NUM2NAME_REL),
            cfg,
        )

    @classmethod
    def predict_with_textrcnn(cls, text, model_info, cfg):
        if not cfg.get('enable_textrcnn', True):
            return {'enabled': False, 'available': False, 'label': None, 'prob': 0.0, 'error': 'textrcnn disabled'}
        # 延迟导入，避免在仅使用传统模型时触发 torch 导入崩溃。
        from algorithm.torch_nlp import predict_textrcnn

        vocab = cfg.get('textrcnn_vocab_path') or cfg.get('textcnn_vocab_path') or textrcnn_vocab_rel()
        weight = cfg.get('textrcnn_weight_path') or (
            model_info.file_path if model_info and model_info.model_type == 'textrcnn' else None
        ) or textrcnn_weight_rel()
        vocab = cls._resolve_existing_rel(vocab, 'model/textcnn_vocab.json')
        weight = cls._resolve_existing_rel(weight, 'model/textrcnn.pt')
        return predict_textrcnn(
            text,
            weight,
            vocab,
            cfg.get('num2name_path', NUM2NAME_REL),
            cfg,
        )

    @classmethod
    def predict_with_rules(cls, text, cfg):
        return predict_rules(text, enabled=cfg.get('enable_rules', True))

    @classmethod
    def fuse_prediction_result(cls, result_map, cfg):
        parts = []
        weight_map = [
            ('svm', 'weight_svm'),
            ('knn', 'weight_knn'),
            ('rf', 'weight_rf'),
            ('dt', 'weight_dt'),
            ('lr', 'weight_lr'),
            ('textcnn', 'weight_textcnn'),
            ('textrcnn', 'weight_textrcnn'),
            ('rules', 'weight_rules'),
        ]
        for key, wkey in weight_map:
            r = result_map.get(key) or {}
            if not r.get('enabled') or not r.get('available'):
                continue
            w = finite_float(cfg.get(wkey, 0))
            if w <= 0:
                continue
            score = finite_float(r.get('prob', r.get('score', 0.0)))
            parts.append((score, w))
        return fuse_weighted_scores(parts, cfg)

    @classmethod
    def _best_label_from_models(cls, result_map):
        candidates = []
        for key in ('svm', 'knn', 'rf', 'dt', 'lr', 'textcnn', 'textrcnn'):
            r = result_map.get(key) or {}
            if r.get('available') and r.get('label'):
                candidates.append((finite_float(r.get('prob', 0)), r.get('label')))
        if not candidates:
            return None
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1]

    @classmethod
    def _max_model_prob(cls, result_map):
        m = 0.0
        for key in ('svm', 'knn', 'rf', 'dt', 'lr', 'textcnn', 'textrcnn'):
            r = result_map.get(key) or {}
            if r.get('available'):
                m = max(m, finite_float(r.get('prob', 0.0)))
        return m

    @classmethod
    def _resolve_label(cls, pred_name, risk_level):
        if pred_name:
            label = LabelDefinition.objects.filter(name=pred_name).first()
            if label:
                return label
        return LabelDefinition.objects.filter(risk_level=risk_level, is_active=True).order_by('code').first()

    @classmethod
    def save_prediction_result(
        cls,
        raw_text,
        model_info,
        predicted_label,
        probability_score,
        rule_score,
        final_risk_score,
        risk_level,
        hit_keywords,
        cfg,
    ):
        alert_threshold = float(cfg.get('threshold_alert', 0.65))
        rule_alert_threshold = float(cfg.get('threshold_rule_alert', 0.5))
        is_alert_triggered = (
            float(final_risk_score) >= alert_threshold
            or int(risk_level) >= 2
            or float(rule_score) >= rule_alert_threshold
        )
        kw_list = hit_keywords if isinstance(hit_keywords, (list, tuple)) else []
        kw_joined = ','.join(str(x) for x in kw_list)
        obj = PredictionResult.objects.create(
            raw_text=raw_text,
            model_info=model_info,
            model=model_info,
            predicted_label=predicted_label,
            probability_score=float(probability_score),
            rule_score=float(rule_score),
            final_risk_score=float(final_risk_score),
            is_alert_triggered=is_alert_triggered,
            probability=float(probability_score),
            risk_score=float(final_risk_score),
            risk_level=int(risk_level),
            hit_keywords=kw_joined[:512],
            detail_json=json.dumps(
                _json_clean_for_detail(
                    {'model_prob': probability_score, 'rule_score': rule_score, 'keywords': kw_list}
                ),
                ensure_ascii=False,
            ),
        )
        return obj

    @classmethod
    def _compute_alert_trigger_reason(cls, prediction_obj, rule_score, hit_keywords):
        if not prediction_obj.is_alert_triggered:
            return ''

        trigger_reason = []
        if prediction_obj.risk_level >= 2 and rule_score < 0.5:
            trigger_reason.append('模型高风险')
        elif prediction_obj.risk_level < 2 and rule_score >= 0.5:
            trigger_reason.append('规则词命中')
        else:
            trigger_reason.append('模型+规则组合')

        if hit_keywords:
            trigger_reason.append('命中词:' + ','.join(hit_keywords))

        return ';'.join(trigger_reason)

    @classmethod
    def build_result_map(cls, text, model_info, cfg):
        return {
            'svm': cls.predict_with_svm(text, model_info, cfg),
            'knn': cls.predict_with_knn(text, model_info, cfg),
            'rf': cls.predict_with_rf(text, model_info, cfg),
            'dt': cls.predict_with_dt(text, model_info, cfg),
            'lr': cls.predict_with_lr(text, model_info, cfg),
            'textcnn': cls.predict_with_textcnn(text, model_info, cfg),
            'textrcnn': cls.predict_with_textrcnn(text, model_info, cfg),
            'rules': cls.predict_with_rules(text, cfg),
        }

    @classmethod
    def predict_text_line(cls, text, model_info=None):
        """单行预测（如前台演示），不写数据库。model_info 为 None 时使用当前启用模型（无则默认 SVM 基线）。"""
        mi = model_info if model_info is not None else cls.load_active_model()
        if not mi:
            raise ValueError('无可用模型')
        if mi.model_type in HF_MODEL_TYPES:
            return cls._predict_text_line_hf(text, mi)
        cfg = cls.get_runtime_config(mi)
        result_map = cls.build_result_map(text, mi, cfg)
        final_score, risk_level = cls.fuse_prediction_result(result_map, cfg)
        pred_name = cls._best_label_from_models(result_map)
        rules = result_map.get('rules') or {}
        return {
            'label': pred_name,
            'final_score': final_score,
            'risk_level': risk_level,
            'rule_score': float(rules.get('score', 0)),
            'submodels': result_map,
            'model_id': mi.id,
            'model_name': mi.name,
            'model_version': mi.version,
            'model_type': mi.model_type,
            'model_type_display': mi.get_model_type_display(),
        }

    @classmethod
    def _hf_db_label_name(cls, model_info, hf_ret):
        pid = hf_ret.get('pred_id')
        if pid is None:
            return None
        if model_info.model_type == 'hf_depr_bin':
            return map_binary_depression_label(int(pid))
        if model_info.model_type == 'hf_mh_4cls':
            lbl = hf_ret.get('raw_label') or ''
            if isinstance(lbl, str) and not lbl.startswith('LABEL_'):
                return {
                    'Anxiety': 'HF_Anxiety',
                    'Depression': 'HF_Depression',
                    'Normal': 'HF_Normal',
                    'Suicidal': 'HF_Suicidal',
                }.get(lbl.strip(), map_four_class_mental_health_label(int(pid)))
            return map_four_class_mental_health_label(int(pid))
        return None

    @classmethod
    def _predict_text_line_hf(cls, text, model_info):
        rel = (model_info.file_path or '').strip()
        hf_ret = predict_hf_sequence_classifier(rel, text)
        if hf_ret.get('error'):
            raise ValueError(hf_ret['error'])
        pred_name = cls._hf_db_label_name(model_info, hf_ret)
        label_obj = LabelDefinition.objects.filter(name=pred_name).first() if pred_name else None
        risk_level = int(label_obj.risk_level) if label_obj else 0
        rules = predict_rules(text, enabled=True)
        rule_score = float(rules.get('score', 0.0))
        prob = float(hf_ret.get('prob') or 0.0)
        final_score = max(prob, rule_score)
        return {
            'label': pred_name,
            'final_score': final_score,
            'risk_level': risk_level,
            'rule_score': rule_score,
            'submodels': {'hf_transformers': {'available': True, 'label': pred_name, 'prob': prob, 'raw': hf_ret}},
            'model_id': model_info.id,
            'model_name': model_info.name,
            'model_version': model_info.version,
            'model_type': model_info.model_type,
            'model_type_display': model_info.get_model_type_display(),
        }

    @classmethod
    def _run_prediction_hf(cls, raw_obj, model_info):
        rel = (model_info.file_path or '').strip()
        hf_ret = predict_hf_sequence_classifier(rel, raw_obj.content)
        if hf_ret.get('error'):
            raise ValueError(hf_ret['error'])
        pred_name = cls._hf_db_label_name(model_info, hf_ret)
        label_obj = LabelDefinition.objects.filter(name=pred_name).first() if pred_name else None
        if not label_obj:
            label_obj = cls._resolve_label(pred_name, 0)

        rules = predict_rules(raw_obj.content, enabled=True)
        hit_keywords = rules.get('hit_words') or []
        rule_score = float(rules.get('score', 0.0))
        probability_score = float(hf_ret.get('prob') or 0.0)
        risk_level = int(label_obj.risk_level) if label_obj else 0
        final_score = max(probability_score, rule_score)

        cfg = cls.get_runtime_config(model_info)
        prediction_obj = cls.save_prediction_result(
            raw_text=raw_obj,
            model_info=model_info,
            predicted_label=label_obj,
            probability_score=probability_score,
            rule_score=rule_score,
            final_risk_score=final_score,
            risk_level=risk_level,
            hit_keywords=hit_keywords,
            cfg=cfg,
        )
        cleaned_text = preprocess_pipeline(raw_obj.content)
        prediction_detail = {
            'raw_text': raw_obj.content,
            'cleaned_text': cleaned_text,
            'model_version': model_info.version,
            'model_name': model_info.name,
            'hf_transformers': hf_ret,
            'rules': rules,
        }
        prediction_obj.detail_json = json.dumps(
            _json_clean_for_detail(prediction_detail), ensure_ascii=False
        )
        prediction_obj.save(update_fields=['detail_json'])

        raw_obj.transit_to(RawText.STATUS_PREDICTED)
        trigger_reason = cls._compute_alert_trigger_reason(prediction_obj, rule_score, hit_keywords)
        if trigger_reason:
            prediction_detail['alert'] = {'is_alert_triggered': True, 'trigger_reason': trigger_reason}
            prediction_obj.detail_json = json.dumps(
                _json_clean_for_detail(prediction_detail), ensure_ascii=False
            )
            prediction_obj.save(update_fields=['detail_json'])
        return prediction_obj, None

    @classmethod
    def run_prediction_for_rawtext(cls, rawtext_id, model_info_id=None):
        raw_obj = RawText.objects.filter(id=rawtext_id).first()
        if not raw_obj:
            raise ValueError('RawText 不存在')

        if raw_obj.status == RawText.STATUS_LABELED:
            raw_obj.transit_to(RawText.STATUS_PENDING_PREDICT)
        if raw_obj.status != RawText.STATUS_PENDING_PREDICT:
            raise ValueError('当前状态不是待预测，无法执行预测: {}'.format(raw_obj.get_status_display()))

        model_info = cls._resolve_model_for_run(model_info_id=model_info_id)
        if model_info.model_type in HF_MODEL_TYPES:
            return cls._run_prediction_hf(raw_obj, model_info)

        cfg = cls.get_runtime_config(model_info)

        result_map = cls.build_result_map(raw_obj.content, model_info, cfg)

        final_score, risk_level = cls.fuse_prediction_result(result_map, cfg)
        pred_name = cls._best_label_from_models(result_map)
        label_obj = cls._resolve_label(pred_name, risk_level)
        hit_keywords = (result_map.get('rules') or {}).get('hit_words', [])
        rule_score = float((result_map.get('rules') or {}).get('score', 0.0))
        probability_score = cls._max_model_prob(result_map)

        prediction_obj = cls.save_prediction_result(
            raw_text=raw_obj,
            model_info=model_info,
            predicted_label=label_obj,
            probability_score=probability_score,
            rule_score=rule_score,
            final_risk_score=final_score,
            risk_level=risk_level,
            hit_keywords=hit_keywords,
            cfg=cfg,
        )

        cleaned_text = preprocess_pipeline(raw_obj.content)
        fusion_detail = {
            'weights': {
                'weight_svm': float(cfg.get('weight_svm', 0)),
                'weight_knn': float(cfg.get('weight_knn', 0)),
                'weight_rf': float(cfg.get('weight_rf', 0)),
                'weight_dt': float(cfg.get('weight_dt', 0)),
                'weight_lr': float(cfg.get('weight_lr', 0)),
                'weight_textcnn': float(cfg.get('weight_textcnn', 0)),
                'weight_textrcnn': float(cfg.get('weight_textrcnn', 0)),
                'weight_rules': float(cfg.get('weight_rules', 0)),
            },
            'thresholds': {
                'high_risk': float(cfg.get('threshold_high_risk', 0.85)),
                'alert': float(cfg.get('threshold_alert', 0.65)),
                'rule_alert': float(cfg.get('threshold_rule_alert', 0.5)),
            },
            'final_risk_score': final_score,
            'risk_level': risk_level,
        }

        prediction_detail = {
            'raw_text': raw_obj.content,
            'cleaned_text': cleaned_text,
            'model_version': model_info.version if model_info else '',
            'model_name': model_info.name if model_info else '',
            'submodels': result_map,
            'fusion': fusion_detail,
        }
        prediction_obj.detail_json = json.dumps(
            _json_clean_for_detail(prediction_detail), ensure_ascii=False
        )
        prediction_obj.save(update_fields=['detail_json'])

        raw_obj.transit_to(RawText.STATUS_PREDICTED)
        trigger_reason = cls._compute_alert_trigger_reason(prediction_obj, rule_score, hit_keywords)
        if trigger_reason:
            prediction_detail['alert'] = {'is_alert_triggered': True, 'trigger_reason': trigger_reason}
            prediction_obj.detail_json = json.dumps(
                _json_clean_for_detail(prediction_detail), ensure_ascii=False
            )
            prediction_obj.save(update_fields=['detail_json'])
        return prediction_obj, None

    @classmethod
    def run_batch_prediction(cls, rawtext_ids, model_info_id=None):
        results = []
        seen = set()
        for rid in rawtext_ids:
            if rid in seen:
                continue
            seen.add(rid)
            try:
                pred_obj, alert_obj = cls.run_prediction_for_rawtext(rid, model_info_id=model_info_id)
                results.append({
                    'rawtext_id': rid,
                    'ok': True,
                    'prediction_id': pred_obj.id,
                    'alert_id': alert_obj.id if alert_obj else None,
                })
            except Exception as e:
                logger.exception('run_prediction_for_rawtext 失败 rawtext_id=%s', rid)
                tb = traceback.format_exc()
                err = str(e) if str(e) else repr(e)
                if tb:
                    err = '{}\n{}'.format(err, tb)
                if len(err) > 8000:
                    err = err[:8000] + '\n…(已截断)'
                results.append({'rawtext_id': rid, 'ok': False, 'error': err})
        return results
