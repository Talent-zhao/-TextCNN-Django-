import datetime as dt
import hashlib

from django.core.management.base import BaseCommand

from app.models import Comment, CommentRawTextMap, RawText, TextSource
from app.services.predict_service import PredictService


class Command(BaseCommand):
    help = '回填历史 Comment 到 RawText/CommentRawTextMap，并可选补跑预测结果'

    def add_arguments(self, parser):
        parser.add_argument('--predict', action='store_true', help='回填后补跑预测（仅 pending_predict）')
        parser.add_argument('--model-id', dest='model_id', required=False, help='可选，指定 ModelInfo 主键')
        parser.add_argument('--batch-size', dest='batch_size', type=int, default=200, help='预测批量大小，默认 200')
        parser.add_argument(
            '--force-remap',
            action='store_true',
            help='默认不覆盖已有 CommentRawTextMap；开启后允许将已有映射改指向回填 RawText',
        )

    def handle(self, *args, **options):
        source, _ = TextSource.objects.get_or_create(
            name='贴吧评论回填',
            source_type='tieba',
            source_key='comment_backfill',
            defaults={'is_enabled': True},
        )

        predict_enabled = bool(options.get('predict'))
        model_id_raw = (options.get('model_id') or '').strip()
        model_info_id = int(model_id_raw) if model_id_raw.isdigit() else None
        batch_size = int(options.get('batch_size') or 200)
        if batch_size <= 0:
            batch_size = 200
        force_remap = bool(options.get('force_remap'))

        created_raw = 0
        reused_raw = 0
        created_map = 0
        updated_map = 0
        skipped_has_map = 0
        pending_ids = []

        for c in Comment.objects.all().order_by('id'):
            existing_map = CommentRawTextMap.objects.filter(comment=c).select_related('raw_text').first()
            if existing_map and not force_remap:
                # 默认行为：已有稳定映射则跳过，避免覆盖 init(tid) 已建立的链路
                skipped_has_map += 1
                continue

            content = (c.content or '').strip()
            if not content:
                continue

            floor = int(c.floor or 0)
            t = c.time.strftime('%Y-%m-%d') if c.time else ''
            uid = c.user_url or ''
            dedup_hash = hashlib.sha256(
                "{}|{}|{}|{}".format(t, floor, uid, content).encode('utf-8')
            ).hexdigest()
            external_id = "comment:{}:{}:{}".format(c.id, floor, dedup_hash[:16])

            publish_time = None
            if c.time:
                publish_time = dt.datetime.combine(c.time, dt.time.min)

            raw_obj = RawText.objects.filter(source=source, external_id=external_id).first()
            if raw_obj is None:
                raw_obj = RawText.objects.create(
                    source=source,
                    external_id=external_id,
                    author_name=c.name or '',
                    content=content,
                    publish_time=publish_time,
                    status=RawText.STATUS_PENDING_PREDICT,
                    dedup_hash=dedup_hash,
                )
                created_raw += 1
            else:
                reused_raw += 1

            if existing_map is None:
                CommentRawTextMap.objects.create(comment=c, raw_text=raw_obj)
                created_map += 1
            elif force_remap and existing_map.raw_text_id != raw_obj.id:
                existing_map.raw_text = raw_obj
                existing_map.save(update_fields=['raw_text'])
                updated_map += 1

            if predict_enabled and raw_obj.status == RawText.STATUS_PENDING_PREDICT:
                pending_ids.append(raw_obj.id)

        predict_ok = 0
        predict_fail = 0
        if predict_enabled and pending_ids:
            # 去重并按批次执行
            pending_ids = list(dict.fromkeys(pending_ids))
            for i in range(0, len(pending_ids), batch_size):
                batch_ids = pending_ids[i:i + batch_size]
                result = PredictService.run_batch_prediction(batch_ids, model_info_id=model_info_id)
                predict_ok += len([x for x in result if x.get('ok')])
                predict_fail += len([x for x in result if not x.get('ok')])

        self.stdout.write(self.style.SUCCESS(
            '回填完成: 新建RawText={} 复用RawText={} 新建映射={} 更新映射={} 已有映射跳过={} 预测成功={} 预测失败={}'.format(
                created_raw, reused_raw, created_map, updated_map, skipped_has_map, predict_ok, predict_fail
            )
        ))
