# -*- coding: utf-8 -*-
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0014_user_is_active_allowed_model_types'),
    ]

    operations = [
        migrations.CreateModel(
            name='FusionConfigPreset',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('is_auto', models.BooleanField(default=False, verbose_name='系统自动快照')),
                ('name', models.CharField(max_length=128, verbose_name='方案名称')),
                ('remark', models.CharField(blank=True, default='', max_length=255, verbose_name='备注')),
                ('snapshot_json', models.TextField(verbose_name='快照数据JSON')),
                ('created_at', models.DateTimeField(auto_now_add=True, verbose_name='创建时间')),
                ('updated_at', models.DateTimeField(auto_now=True, verbose_name='更新时间')),
            ],
            options={
                'verbose_name': '融合配置快照',
                'verbose_name_plural': '融合配置快照',
                'ordering': ['-updated_at'],
            },
        ),
    ]
