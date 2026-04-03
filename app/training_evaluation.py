# -*- coding: utf-8 -*-
"""
训练结束后将验证集指标写入 ModelEvaluation（供后台「评估结果」查看）。
由 scripts 在成功写盘后调用；失败只打日志，不影响训练退出码。

同时在传入权重路径时，按验证集 macro F1 不劣于已登记最优的原则，更新对应 model_type 下
「源算法模型」登记的 file_path / vectorizer_path（不会写入名称「融合策略」的登记行）。

传入 experiment_csv_rel 时，会追加一条 AlgorithmExperimentRecord（算法对比实验记录）。
跳过：环境变量 DEPRESSION_WEB_SKIP_TRAIN_EXPERIMENT_RECORD=1。
"""
import json
import os
import sys


def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _setup_django():
    root = _project_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'DjangoWeb.settings')
    import django

    django.setup()


def roc_auc_from_proba(y_true, y_proba):
    """
    由验证集真实标签与 predict_proba / softmax 概率计算 ROC AUC。
    - 二分类：正类取第 2 列概率；
    - 多分类：multi_class='ovr' + average='macro'。
    无法计算（单类、形状不符等）时返回 None。
    """
    try:
        import numpy as np
        from sklearn.metrics import roc_auc_score
    except Exception:
        return None

    try:
        y_true = np.asarray(y_true).astype(int).ravel()
        y_proba = np.asarray(y_proba, dtype=float)
    except Exception:
        return None
    if y_proba.ndim != 2 or y_proba.shape[0] != len(y_true):
        return None
    n_classes = y_proba.shape[1]
    if n_classes < 2:
        return None
    uniq = np.unique(y_true)
    if len(uniq) < 2:
        return None
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, y_proba[:, 1]))
        labels = list(range(n_classes))
        return float(
            roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro', labels=labels)
        )
    except Exception:
        return None


def infer_overfitting_from_train_val_acc(train_acc, val_acc, gap_threshold=0.06):
    """
    用「训练集准确率 − 验证集准确率」判断是否过拟合。
    训练明显高于验证（差距 > gap_threshold，默认 6%）时返回 True，否则 False。
    输入非法时返回 None。
    """
    try:
        ta = float(train_acc)
        va = float(val_acc)
    except (TypeError, ValueError):
        return None
    if ta < 0 or ta > 1.0001 or va < 0 or va > 1.0001:
        return None
    gap = ta - va
    return bool(gap > gap_threshold)


def metrics_from_classification_report_dict(rep_dict, y_true, y_pred):
    """验证集准确率 + classification_report(..., output_dict=True) 的 macro avg 三项。"""
    from sklearn.metrics import accuracy_score

    acc = float(accuracy_score(y_true, y_pred))
    ma = rep_dict.get('macro avg') or {}

    def _f(x):
        try:
            return float(x) if x is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    p = _f(ma.get('precision'))
    r = _f(ma.get('recall'))
    f1 = _f(ma.get('f1-score'))
    return acc, p, r, f1


def _registered_best_f1(model_info):
    """从 ModelInfo.metrics_json 读取历史登记最优 macro F1；无则 None。"""
    raw = (model_info.metrics_json or '').strip()
    if not raw:
        return None
    try:
        j = json.loads(raw)
        v = j.get('best_val_f1')
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _should_update_registered_paths(model_info, new_f1):
    """
    登记路径仅在「不劣于」历史最优 macro F1 时更新（持平则刷新为本轮产物）。
    尚无路径或尚无 best 记录时一律更新。
    """
    if not (model_info.file_path or '').strip():
        return True
    old = _registered_best_f1(model_info)
    if old is None:
        return True
    try:
        return float(new_f1) >= float(old) - 1e-9
    except (TypeError, ValueError):
        return True


def _truncate_field(s, max_len):
    s = (s or '').strip().replace('\\', '/')
    if not s:
        return ''
    return s[:max_len] if len(s) > max_len else s


def _persist_algorithm_experiment_record(
    algo_type,
    val_accuracy,
    precision,
    recall,
    f1_score,
    auc_v,
    train_v,
    is_overfitting,
    eval_version,
    dataset_version,
    train_sample_scale,
    sklearn_max_features,
    hyperparams_extra,
):
    """训练成功后追加一条「算法对比实验记录」。"""
    if os.environ.get('DEPRESSION_WEB_SKIP_TRAIN_EXPERIMENT_RECORD', '').strip().lower() in (
        '1',
        'true',
        'yes',
    ):
        return

    from app.models import AlgorithmExperimentRecord

    valid_algo = {c[0] for c in AlgorithmExperimentRecord.ALGO_TYPE_CHOICES}
    if algo_type not in valid_algo:
        return

    valid_s = {c[0] for c in AlgorithmExperimentRecord.TRAIN_SAMPLE_CHOICES}
    ts = (train_sample_scale or 'full').strip().lower()
    if ts not in valid_s:
        ts = 'full'

    valid_m = {c[0] for c in AlgorithmExperimentRecord.SKLEARN_MAX_FEATURES_CHOICES}
    mf = (sklearn_max_features or '8000').strip().lower()
    if mf not in valid_m:
        mf = '8000'

    ds = _truncate_field(dataset_version, 64)
    if not ds:
        ds = 'default'

    hp = {
        'eval_version': (eval_version or '')[:32],
        'source': 'train_scripts',
    }
    if is_overfitting is True:
        hp['is_overfitting'] = True
    elif is_overfitting is False:
        hp['is_overfitting'] = False
    if isinstance(hyperparams_extra, dict):
        hp.update(hyperparams_extra)

    try:
        hp_txt = json.dumps(hp, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        hp_txt = '{}'

    name = '训练自动生成 · {}'.format(algo_type.upper())
    name = _truncate_field(name, 128)

    remark = ''
    if eval_version:
        remark = 'eval:{}'.format(eval_version[:32])
    remark = _truncate_field(remark, 255)

    AlgorithmExperimentRecord.objects.create(
        name=name,
        dataset_version=ds,
        algorithm_type=algo_type,
        train_sample_scale=ts,
        sklearn_max_features=mf,
        hyperparams_json=hp_txt or '{}',
        training_time_sec=train_v,
        accuracy=float(val_accuracy),
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1_score),
        auc=auc_v,
        remark=remark,
    )
    print(
        '[experiment_record] 已追加算法对比实验记录: {} / 数据:{} / F1={:.4f}'.format(
            name,
            ds,
            float(f1_score),
        ),
        flush=True,
    )


def save_evaluation_from_training(
    algo_type,
    val_accuracy,
    precision,
    recall,
    f1_score,
    auc=None,
    training_time_sec=None,
    is_overfitting=None,
    file_path=None,
    vectorizer_path=None,
    experiment_csv_rel=None,
    experiment_train_sample_scale=None,
    experiment_sklearn_max_features=None,
    experiment_hyperparams_extra=None,
):
    """
    algo_type: svm | knn | rf | dt | lr | textcnn | textrcnn
    各指标为 0～1（验证集 / 划分出的验证子集）。
    auc、training_time_sec、is_overfitting 可选，供算法对比页展示。
    file_path / vectorizer_path: 相对项目根的 POSIX 路径；传入且 macro F1 不劣于已登记时写入 ModelInfo。
    experiment_*: 若提供 experiment_csv_rel，则同步写入 AlgorithmExperimentRecord。
    """
    if os.environ.get('DEPRESSION_WEB_SKIP_TRAIN_EVAL', '').strip().lower() in ('1', 'true', 'yes'):
        return
    try:
        _setup_django()
    except Exception as e:
        print('[eval_persist] 跳过写入评估（Django 未就绪）:', e, flush=True)
        return
    try:
        from django.utils import timezone

        from app.models import ModelEvaluation, ModelInfo
        from app.registry_names import (
            LEGACY_SOURCE_ALGO_NAMES,
            SOURCE_ALGO_MODELINFO_NAME,
            fusion_strategy_modelinfo_name_set,
            source_algo_modelinfo_name_set,
        )

        algo_type = (algo_type or '').strip().lower()
        if not algo_type:
            return

        fusion_names = fusion_strategy_modelinfo_name_set()
        source_names = source_algo_modelinfo_name_set()

        mi = (
            ModelInfo.objects.filter(model_type=algo_type, name__in=source_names)
            .order_by('-is_active', '-id')
            .first()
        )
        if not mi:
            mi = (
                ModelInfo.objects.filter(model_type=algo_type)
                .exclude(name__in=fusion_names)
                .order_by('-is_active', '-id')
                .first()
            )
        if not mi:
            mi, _ = ModelInfo.objects.get_or_create(
                name=SOURCE_ALGO_MODELINFO_NAME,
                version=algo_type,
                defaults={
                    'model_type': algo_type,
                    'status': 'ready',
                    'is_active': False,
                    'file_path': '',
                    'vectorizer_path': '',
                    'description': '训练脚本自动写入评估记录时使用的关联项（可按需在模型管理中调整）。',
                    'config_json': '{}',
                    'metrics_json': '{}',
                },
            )
        elif mi.name in LEGACY_SOURCE_ALGO_NAMES:
            mi.name = SOURCE_ALGO_MODELINFO_NAME
            mi.save(update_fields=['name'])
# 项目原作者：赵有才 
# 联系方式：creepreme@126.com/ wechat: zyb1209121xty

        ver = 'train_{}'.format(timezone.now().strftime('%Y%m%d_%H%M%S'))
        auc_v = None
        if auc is not None:
            try:
                auc_v = float(auc)
            except (TypeError, ValueError):
                auc_v = None
        train_v = None
        if training_time_sec is not None:
            try:
                train_v = float(training_time_sec)
            except (TypeError, ValueError):
                train_v = None
        ModelEvaluation.objects.create(
            model_info=mi,
            model_version=ver[:32],
            accuracy=float(val_accuracy),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1_score),
            auc=auc_v,
            training_time_sec=train_v,
            is_overfitting=is_overfitting,
            is_active_snapshot=bool(mi.is_active),
        )

        fp = (file_path or '').strip().replace('\\', '/')
        vp = (vectorizer_path or '').strip().replace('\\', '/') if vectorizer_path else ''
        new_f1 = float(f1_score)
        if fp:
            if _should_update_registered_paths(mi, new_f1):
                mi.file_path = fp
                if vp:
                    mi.vectorizer_path = vp
                snap = {
                    'best_val_f1': float(f1_score),
                    'best_val_accuracy': float(val_accuracy),
                    'best_precision': float(precision),
                    'best_recall': float(recall),
                    'eval_version': ver[:32],
                    'best_recorded_at': timezone.now().isoformat(),
                }
                if auc_v is not None:
                    snap['best_auc'] = float(auc_v)
                mi.metrics_json = json.dumps(snap, ensure_ascii=False, indent=2)
                mi.save(update_fields=['file_path', 'vectorizer_path', 'metrics_json'])
                print(
                    '[eval_persist] 已更新「源算法模型」登记路径（macro F1 不劣于历史）file_path={} vectorizer_path={}'.format(
                        fp, vp or '(未传)',
                    ),
                    flush=True,
                )
            else:
                old_b = _registered_best_f1(mi)
                print(
                    '[eval_persist] 本轮 macroF1={:.4f} 低于已登记最优 {:.4f}，保留原文件路径'.format(
                        new_f1, old_b if old_b is not None else float('nan'),
                    ),
                    flush=True,
                )

        extra = ''
        if auc_v is not None:
            extra += ' AUC={:.4f}'.format(auc_v)
        if train_v is not None:
            extra += ' train_sec={:.1f}'.format(train_v)
        if is_overfitting is True:
            extra += ' 过拟合=是'
        elif is_overfitting is False:
            extra += ' 过拟合=否'
        print(
            '[eval_persist] 已写入「评估结果」: {} 验证准确率={:.4f} macroP/R/F1={:.4f}/{:.4f}/{:.4f}{}'.format(
                algo_type, val_accuracy, precision, recall, f1_score, extra
            ),
            flush=True,
        )

        if (experiment_csv_rel or '').strip():
            try:
                _persist_algorithm_experiment_record(
                    algo_type,
                    val_accuracy,
                    precision,
                    recall,
                    f1_score,
                    auc_v,
                    train_v,
                    is_overfitting,
                    ver[:32],
                    experiment_csv_rel,
                    experiment_train_sample_scale,
                    experiment_sklearn_max_features,
                    experiment_hyperparams_extra,
                )
            except Exception as ex_r:
                print('[experiment_record] 写入实验记录失败（可忽略）:', repr(ex_r), flush=True)
    except Exception as e:
        print('[eval_persist] 写入评估失败（训练仍视为成功）:', repr(e), flush=True)
