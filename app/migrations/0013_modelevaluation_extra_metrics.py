# -*- coding: utf-8 -*-
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0012_remove_alert_center'),
    ]

    operations = [
        migrations.AddField(
            model_name='modelevaluation',
            name='auc',
            field=models.FloatField(blank=True, null=True, verbose_name='AUC'),
        ),
        migrations.AddField(
            model_name='modelevaluation',
            name='training_time_sec',
            field=models.FloatField(blank=True, null=True, verbose_name='训练耗时(秒)'),
        ),
        migrations.AddField(
            model_name='modelevaluation',
            name='is_overfitting',
            field=models.BooleanField(blank=True, null=True, verbose_name='是否过拟合'),
        ),
    ]
