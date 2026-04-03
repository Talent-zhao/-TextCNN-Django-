# Generated manually for extended ModelInfo.model_type choices

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0010_user_role'),
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
                ],
                max_length=16,
                verbose_name='模型类型',
            ),
        ),
    ]
