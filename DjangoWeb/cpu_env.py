# -*- coding: utf-8 -*-
"""限制 OpenMP/BLAS 线程数，降低 Windows 上多库同时加载时进程异常退出的概率。"""
import os


def apply_default_cpu_thread_env():
    pairs = (
        ('OMP_NUM_THREADS', '1'),
        ('MKL_NUM_THREADS', '1'),
        ('OPENBLAS_NUM_THREADS', '1'),
        ('NUMEXPR_NUM_THREADS', '1'),
        ('VECLIB_MAXIMUM_THREADS', '1'),
    )
    for key, val in pairs:
        os.environ.setdefault(key, val)
