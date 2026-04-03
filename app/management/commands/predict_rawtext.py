# -*- coding: utf-8 -*-
"""在独立进程中执行待预测 RawText，避免在主 runserver 进程内加载 torch 导致进程退出。"""
import json
import sys

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = '批量执行 RawText 预测；仅向 stdout 输出一行 JSON（供后台子进程解析）。'

    def add_arguments(self, parser):
        parser.add_argument(
            '--ids',
            dest='ids',
            required=True,
            help='逗号分隔的 RawText 主键，例如 1,2,3',
        )
        parser.add_argument(
            '--model-id',
            dest='model_id',
            required=False,
            help='可选，指定使用的 ModelInfo 主键',
        )

    def handle(self, *args, **options):
        from app.services.predict_service import PredictService

        raw = (options['ids'] or '').strip()
        ids = []
        for part in raw.split(','):
            part = part.strip()
            if part.isdigit():
                ids.append(int(part))
        ids = list(dict.fromkeys(ids))
        if not ids:
            sys.stdout.write(json.dumps({'result': [], 'success_count': 0, 'fail_count': 0, 'error': '无有效 id'}, ensure_ascii=False))
            sys.stdout.write('\n')
            sys.stdout.flush()
            sys.exit(1)

        model_id_raw = (options.get('model_id') or '').strip()
        model_info_id = None
        if model_id_raw:
            if not model_id_raw.isdigit():
                sys.stdout.write(
                    json.dumps(
                        {'result': [], 'success_count': 0, 'fail_count': 0, 'error': 'model_id 非法'},
                        ensure_ascii=False,
                    )
                )
                sys.stdout.write('\n')
                sys.stdout.flush()
                sys.exit(1)
            model_info_id = int(model_id_raw)

        result = PredictService.run_batch_prediction(ids, model_info_id=model_info_id)
        success_count = sum(1 for i in result if i.get('ok'))
        payload = {
            'result': result,
            'success_count': success_count,
            'fail_count': len(result) - success_count,
        }
        sys.stdout.write(json.dumps(payload, ensure_ascii=False))
        sys.stdout.write('\n')
        sys.stdout.flush()
