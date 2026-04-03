# -*- coding: utf-8 -*-
"""
清空管理员端「数据/标注/预测」流水线中的文本相关数据（不含前台 Comment、用户、模型配置）。

用法:
  python manage.py clear_admin_pipeline_data --yes
"""
from django.core.management.base import BaseCommand
from django.db import transaction

from app.models import (
    AnnotationRecord,
    CleanText,
    ExportLog,
    PredictionResult,
    RawText,
)


class Command(BaseCommand):
    help = '删除 RawText 及关联的清洗、标注、预测、导出日志（后台文本数据）'

    def add_arguments(self, parser):
        parser.add_argument(
            '--yes',
            action='store_true',
            help='确认执行，否则不删除',
        )

    def handle(self, *args, **options):
        if not options['yes']:
            self.stderr.write(self.style.WARNING('未执行。若确认清空，请追加参数: --yes'))
            return

        with transaction.atomic():
            n_pred = PredictionResult.objects.count()
            PredictionResult.objects.all().delete()
            n_ann = AnnotationRecord.objects.count()
            AnnotationRecord.objects.all().delete()
            n_cln = CleanText.objects.count()
            CleanText.objects.all().delete()
            n_raw = RawText.objects.count()
            RawText.objects.all().delete()
            n_exp = ExportLog.objects.count()
            ExportLog.objects.all().delete()

        self.stdout.write(
            self.style.SUCCESS(
                '已删除条数 — 预测:{} 标注:{} 清洗:{} 原始文本:{} 导出日志:{}'.format(
                    n_pred, n_ann, n_cln, n_raw, n_exp
                )
            )
        )
        self.stdout.write('保留：LabelDefinition、TextSource、ModelInfo、SystemConfig、Comment（前台）等。')
