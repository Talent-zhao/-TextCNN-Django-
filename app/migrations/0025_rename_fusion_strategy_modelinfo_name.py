# -*- coding: utf-8 -*-

from django.db import migrations


def forwards(apps, schema_editor):
    ModelInfo = apps.get_model('app', 'ModelInfo')
    ModelInfo.objects.filter(name='模型融合策略模型').update(name='融合策略')


class Migration(migrations.Migration):
    dependencies = [
        ('app', '0024_modelinfo_registry_display_names'),
    ]

    operations = [
        migrations.RunPython(forwards, migrations.RunPython.noop),
    ]
