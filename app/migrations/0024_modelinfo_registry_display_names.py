# -*- coding: utf-8 -*-

from django.db import migrations


def forwards(apps, schema_editor):
    ModelInfo = apps.get_model('app', 'ModelInfo')
    ModelInfo.objects.filter(name='训练自动评估').update(name='源算法模型')
    ModelInfo.objects.filter(name='融合配置同步').update(name='模型融合策略模型')


class Migration(migrations.Migration):
    dependencies = [
        ('app', '0023_algorithm_experiment_train_options'),
    ]

    operations = [
        migrations.RunPython(forwards, migrations.RunPython.noop),
    ]
