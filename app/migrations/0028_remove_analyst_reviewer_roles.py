# -*- coding: utf-8 -*-

from django.db import migrations, models


def forwards(apps, schema_editor):
    User = apps.get_model('app', 'User')
    User.objects.filter(role__in=['analyst', 'reviewer']).update(role='user')


class Migration(migrations.Migration):
    dependencies = [
        ('app', '0027_fusion_modelinfo_listed_for_users'),
    ]

    operations = [
        migrations.RunPython(forwards, migrations.RunPython.noop),
        migrations.AlterField(
            model_name='user',
            name='role',
            field=models.CharField(
                choices=[('admin', '管理员'), ('user', '普通用户')],
                default='user',
                max_length=16,
                verbose_name='角色',
            ),
        ),
    ]

