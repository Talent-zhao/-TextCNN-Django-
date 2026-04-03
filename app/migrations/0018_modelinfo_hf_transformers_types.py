# -*- coding: utf-8 -*-
from django.db import migrations, models


def seed_hf_labels_and_models(apps, schema_editor):
    LabelDefinition = apps.get_model('app', 'LabelDefinition')
    ModelInfo = apps.get_model('app', 'ModelInfo')
    rows = [
        (100, 'HF_NotDepressed', 0, 'HF 二分类：非抑郁'),
        (101, 'HF_Depressed', 2, 'HF 二分类：抑郁相关（非临床）'),
        (102, 'HF_Anxiety', 1, 'HF 四分类：焦虑'),
        (103, 'HF_Depression', 2, 'HF 四分类：抑郁'),
        (104, 'HF_Normal', 0, 'HF 四分类：正常'),
        (105, 'HF_Suicidal', 3, 'HF 四分类：自杀倾向（非临床）'),
    ]
    for code, name, risk_level, desc in rows:
        LabelDefinition.objects.get_or_create(
            code=code,
            defaults={
                'name': name,
                'risk_level': risk_level,
                'description': desc[:255],
                'is_active': True,
            },
        )
    ModelInfo.objects.get_or_create(
        name='HF-抑郁二分类',
        version='TRT1000-v1',
        defaults={
            'model_type': 'hf_depr_bin',
            'file_path': 'Best_Modle/binary_depression_distilbert',
            'vectorizer_path': '',
            'status': 'ready',
            'is_active': False,
            'listed_for_users': True,
            'description': 'DistilBERT 英文弱标注抑郁二分类，本地 Best_Modle',
            'config_json': '{}',
            'metrics_json': '{}',
        },
    )
    ModelInfo.objects.get_or_create(
        name='HF-心理健康四分类',
        version='ourafla-v1',
        defaults={
            'model_type': 'hf_mh_4cls',
            'file_path': 'Best_Modle/four_class_mental_health_bert',
            'vectorizer_path': '',
            'status': 'ready',
            'is_active': False,
            'listed_for_users': True,
            'description': 'MentalBERT 微调四分类（Anxiety/Depression/Normal/Suicidal），本地 Best_Modle',
            'config_json': '{}',
            'metrics_json': '{}',
        },
    )


def noop_reverse(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0017_modelinfo_listed_for_users'),
    ]

    operations = [
        migrations.AlterField(
            model_name='modelinfo',
            name='model_type',
            field=models.CharField(
                choices=[
                    ('svm', 'TF-IDF + SVM'),
                    ('knn', 'TF-IDF + KNN（最近邻）'),
                    ('rf', 'TF-IDF + RandomForest（随机森林）'),
                    ('dt', 'TF-IDF + DecisionTree（决策树）'),
                    ('lr', 'TF-IDF + LogisticRegression'),
                    ('textcnn', 'TextCNN'),
                    ('textrcnn', 'TextRCNN'),
                    ('rule', '规则词典'),
                    ('fusion', '融合模型'),
                    ('hf_depr_bin', 'HF DistilBERT 抑郁二分类'),
                    ('hf_mh_4cls', 'HF BERT 心理健康四分类'),
                ],
                max_length=16,
                verbose_name='模型类型',
            ),
        ),
        migrations.RunPython(seed_hf_labels_and_models, noop_reverse),
    ]
