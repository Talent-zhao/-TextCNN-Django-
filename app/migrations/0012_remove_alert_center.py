# -*- coding: utf-8 -*-
from django.db import migrations, models


def migrate_alerted_rawtext(apps, schema_editor):
    RawText = apps.get_model('app', 'RawText')
    RawText.objects.filter(status='alerted').update(status='predicted')


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0011_modelinfo_algorithm_types'),
    ]

    operations = [
        migrations.RunPython(migrate_alerted_rawtext, migrations.RunPython.noop),
        migrations.DeleteModel(name='ReviewRecord'),
        migrations.DeleteModel(name='AlertRecord'),
        migrations.AlterField(
            model_name='rawtext',
            name='status',
            field=models.CharField(
                choices=[
                    ('pending_clean', '未清洗'),
                    ('cleaned', '已清洗'),
                    ('pending_label', '待标注'),
                    ('labeled', '已标注'),
                    ('pending_predict', '待预测'),
                    ('predicted', '已预测'),
                    ('reviewed', '已复核'),
                ],
                default='pending_clean',
                max_length=16,
                verbose_name='处理状态',
            ),
        ),
        migrations.AlterField(
            model_name='exportlog',
            name='export_type',
            field=models.CharField(
                choices=[
                    ('prediction_result', '预测结果'),
                    ('self_check_history', '自检历史'),
                ],
                max_length=32,
                verbose_name='导出类型',
            ),
        ),
    ]
