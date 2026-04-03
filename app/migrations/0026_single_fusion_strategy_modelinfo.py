# -*- coding: utf-8 -*-
"""融合策略改为单条 model_type=fusion、version=default；删除按算法拆分的同名登记。"""

from django.db import migrations


def forwards(apps, schema_editor):
    ModelInfo = apps.get_model('app', 'ModelInfo')
    name = '融合策略'
    version = 'default'
    mtype = 'fusion'

    active_any = ModelInfo.objects.filter(name=name, is_active=True).exists()
    listed_any = ModelInfo.objects.filter(name=name, listed_for_users=True).exists()

    ModelInfo.objects.filter(name=name).exclude(model_type=mtype, version=version).delete()

    fusion, _created = ModelInfo.objects.get_or_create(
        name=name,
        version=version,
        defaults={
            'model_type': mtype,
            'status': 'ready',
            'is_active': active_any,
            'listed_for_users': listed_any,
            'file_path': None,
            'vectorizer_path': None,
            'description': '单条登记：汇总「预测融合设置」中的子模型路径与各算法权重。',
            'config_json': '{}',
            'metrics_json': '{}',
        },
    )
    update_fields = []
    if fusion.model_type != mtype:
        fusion.model_type = mtype
        update_fields.append('model_type')
    if active_any:
        fusion.is_active = True
        update_fields.append('is_active')
    if listed_any:
        fusion.listed_for_users = True
        update_fields.append('listed_for_users')
    if update_fields:
        fusion.save(update_fields=update_fields)


class Migration(migrations.Migration):
    dependencies = [
        ('app', '0025_rename_fusion_strategy_modelinfo_name'),
    ]

    operations = [
        migrations.RunPython(forwards, migrations.RunPython.noop),
    ]
