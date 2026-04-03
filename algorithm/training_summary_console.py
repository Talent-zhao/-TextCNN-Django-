# -*- coding: utf-8 -*-
"""训练结束后控制台摘要，与后台「算法实验记录」表单字段对齐，便于复制。"""
from __future__ import print_function

import json


def _fmt_metric(x, ndigits=4):
    if x is None:
        return '-'
    try:
        return ('{:.' + str(ndigits) + 'f}').format(float(x))
    except (TypeError, ValueError):
        return '-'


def print_experiment_form_footer(
    hyperparams,
    training_time_sec,
    auc,
    accuracy,
    precision,
    recall,
    f1,
):
    """
    在训练脚本收尾打印与「实验记录」中一致的指标块。
    标量行采用「标签：数值」；JSON 仍为「超参数 JSON：」下一行起多行正文。
    hyperparams: 可序列化为 dict 的对象；异常时打印 {}。
    """
    hp = hyperparams if isinstance(hyperparams, dict) else {}
    try:
        hp_txt = json.dumps(hp, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        hp_txt = '{}'

    print('', flush=True)
    print('========== 本轮训练摘要（可对照「实验记录」表单填写）==========', flush=True)
    print('超参数 JSON：', flush=True)
    print(hp_txt, flush=True)
    if training_time_sec is None:
        t_str = '-'
    else:
        try:
            t_str = '{:.3f}'.format(float(training_time_sec))
        except (TypeError, ValueError):
            t_str = '-'
    print('训练时间(秒)：{}'.format(t_str), flush=True)
    print('AUC：{}'.format(_fmt_metric(auc, 4)), flush=True)
    print('Accuracy：{}'.format(_fmt_metric(accuracy, 4)), flush=True)
    print('Precision：{}'.format(_fmt_metric(precision, 4)), flush=True)
    print('Recall：{}'.format(_fmt_metric(recall, 4)), flush=True)
    print('F1：{}'.format(_fmt_metric(f1, 4)), flush=True)
    print('==============================================================', flush=True)
