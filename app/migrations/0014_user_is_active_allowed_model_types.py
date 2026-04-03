# -*- coding: utf-8 -*-
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0013_modelevaluation_extra_metrics'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='is_active',
            field=models.BooleanField(default=True, verbose_name='账号启用'),
        ),
        migrations.AddField(
            model_name='user',
            name='allowed_model_types',
            field=models.TextField(
                blank=True,
                default='',
                help_text='JSON 数组，如 ["svm","textcnn"]；留空表示可使用全部已登记模型类型',
                verbose_name='可用算法类型',
            ),
        ),
        migrations.AddField(
            model_name='user',
            name='admin_note',
            field=models.CharField(blank=True, max_length=255, null=True, verbose_name='管理员备注'),
        ),
    ]
