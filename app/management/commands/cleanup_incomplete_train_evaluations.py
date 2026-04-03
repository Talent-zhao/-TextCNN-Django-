# -*- coding: utf-8 -*-
"""
删除「源算法模型」登记下数据不完整的旧 ModelEvaluation：版本号形如 train_YYYYMMDD_hhmmss，
且尚未写入 AUC 与训练耗时（均为空）的 ModelEvaluation。

用法:
  python manage.py cleanup_incomplete_train_evaluations
  python manage.py cleanup_incomplete_train_evaluations --dry-run
"""
from django.core.management.base import BaseCommand

from app.models import ModelEvaluation


class Command(BaseCommand):
    help = '删除训练自动写入但缺少 AUC 与训练耗时的评估记录（train_* 版本号）'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='只统计与列出，不执行删除',
        )

    def handle(self, *args, **options):
        qs = ModelEvaluation.objects.filter(
            model_version__startswith='train_',
            auc__isnull=True,
            training_time_sec__isnull=True,
        ).select_related('model_info')

        n = qs.count()
        if options['dry_run']:
            self.stdout.write(self.style.WARNING('dry-run：将删除 {} 条'.format(n)))
            for r in qs.order_by('-evaluated_at')[:80]:
                mi = r.model_info
                self.stdout.write(
                    '  id={} ver={} type={}/{}'.format(
                        r.id,
                        r.model_version,
                        mi.name if mi else '-',
                        mi.get_model_type_display() if mi else '-',
                    )
                )
            if n > 80:
                self.stdout.write('  ... 共 {} 条'.format(n))
            return

        deleted, _ = qs.delete()
        self.stdout.write(
            self.style.SUCCESS('已删除不完整训练评估记录 {} 条（ModelEvaluation）'.format(deleted))
        )
