# -*- coding: utf-8 -*-
"""融合策略登记默认可出现在前台预测，便于套餐勾选 fusion 后用户能选到。"""

from django.db import migrations


def forwards(apps, schema_editor):
    ModelInfo = apps.get_model('app', 'ModelInfo')
    ModelInfo.objects.filter(model_type='fusion', status='ready').update(listed_for_users=True)


class Migration(migrations.Migration):
    dependencies = [
        ('app', '0026_single_fusion_strategy_modelinfo'),
    ]

    operations = [
        migrations.RunPython(forwards, migrations.RunPython.noop),
    ]
