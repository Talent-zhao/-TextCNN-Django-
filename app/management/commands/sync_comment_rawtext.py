import hashlib
from datetime import datetime

from django.core.management.base import BaseCommand

from app.models import AnnotationRecord, Comment, LabelDefinition, RawText, TextSource


EMOTION_TO_LABEL = {
    '没有抑郁倾向': ('正常文本', 0),
    '一般抑郁倾向': ('轻度负向情绪', 1),
    '轻度抑郁倾向': ('中风险心理异常倾向', 2),
    '严重抑郁倾向': ('高风险危机表达', 3),
}


def ensure_4class_labels():
    presets = [
        (0, '正常文本', 0, '无明显风险'),
        (1, '轻度负向情绪', 1, '轻度负向表达'),
        (2, '中风险心理异常倾向', 2, '需重点关注'),
        (3, '高风险危机表达', 3, '高危表达，需尽快复核'),
    ]
    for code, name, risk, desc in presets:
        LabelDefinition.objects.update_or_create(
            code=code,
            defaults={'name': name, 'risk_level': risk, 'description': desc, 'is_active': True}
        )


class Command(BaseCommand):
    help = '将旧 Comment 数据兼容同步到 RawText（可选同步历史 emotion 到标注记录）'

    def add_arguments(self, parser):
        parser.add_argument('--sync-label', action='store_true', help='把 Comment.emotion 同步为 AnnotationRecord')

    def handle(self, *args, **options):
        source, _ = TextSource.objects.get_or_create(
            name='贴吧历史评论',
            defaults={'source_type': 'tieba', 'source_key': 'legacy_comment'}
        )
        ensure_4class_labels()

        synced = 0
        skipped = 0
        labeled = 0

        for c in Comment.objects.all().order_by('id'):
            content = (c.content or '').strip()
            if not content:
                skipped += 1
                continue
            dedup_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            raw, created = RawText.objects.get_or_create(
                dedup_hash=dedup_hash,
                defaults={
                    'source': source,
                    'external_id': f'comment_{c.id}',
                    'author_name': c.name,
                    'content': content,
                    'publish_time': datetime.combine(c.time, datetime.min.time()) if c.time else None,
                    'status': RawText.STATUS_PENDING_CLEAN,
                }
            )
            if created:
                synced += 1
            else:
                skipped += 1

            if options['sync_label'] and c.emotion in EMOTION_TO_LABEL:
                label_name, _ = EMOTION_TO_LABEL[c.emotion]
                label = LabelDefinition.objects.filter(name=label_name).first()
                if label:
                    AnnotationRecord.objects.get_or_create(
                        raw_text=raw,
                        label=label,
                        defaults={'dataset_split': 'train', 'remark': '由历史Comment情感迁移'}
                    )
                    if raw.status == RawText.STATUS_PENDING_CLEAN:
                        raw.status = RawText.STATUS_CLEANED
                        raw.save(update_fields=['status', 'updated_at'])
                    if raw.status == RawText.STATUS_CLEANED:
                        raw.transit_to(RawText.STATUS_PENDING_LABEL)
                    if raw.status == RawText.STATUS_PENDING_LABEL:
                        raw.transit_to(RawText.STATUS_LABELED)
                        raw.transit_to(RawText.STATUS_PENDING_PREDICT)
                    labeled += 1

        self.stdout.write(self.style.SUCCESS(
            f'同步完成：新增RawText {synced} 条，跳过 {skipped} 条，迁移标注 {labeled} 条'
        ))
