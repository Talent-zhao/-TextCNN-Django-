from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False, verbose_name='id')),
                ('time', models.DateField(auto_now_add=True, verbose_name='创建时间')),
                ('email', models.EmailField(blank=True, max_length=254, null=True, verbose_name='email')),
                ('username', models.CharField(max_length=64, verbose_name='昵称')),
                ('tel', models.CharField(max_length=64, verbose_name='手机号')),
                ('pwd', models.CharField(max_length=64, verbose_name='密码')),
                ('sex', models.CharField(blank=True, choices=[('男', '男'), ('女', '女')], max_length=4, null=True, verbose_name='性别')),
                ('address', models.CharField(blank=True, max_length=128, null=True, verbose_name='地址')),
                ('avatar', models.ImageField(blank=True, null=True, upload_to='avatar/', verbose_name='用户头像')),
            ],
            options={
                'verbose_name': '用户表',
                'verbose_name_plural': '用户表',
            },
        ),
    ]
