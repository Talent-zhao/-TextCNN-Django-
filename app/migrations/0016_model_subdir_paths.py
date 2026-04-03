# -*- coding: utf-8 -*-
"""将 SystemConfig / ModelInfo 中扁平 model/ 路径更新为按算法子目录存放后的默认路径。"""

from django.db import migrations


def _migrate(apps, schema_editor):
    SystemConfig = apps.get_model('app', 'SystemConfig')
    ModelInfo = apps.get_model('app', 'ModelInfo')

    cfg_rows = [
        ('tfidf_vectorizer_path', 'model/tfidfVectorizer.pkl', 'model/svm/tfidfVectorizer.pkl'),
        ('svm_model_path', 'model/model_svm.pkl', 'model/svm/model_svm.pkl'),
        ('knn_model_path', 'model/model_knn.pkl', 'model/knn/model_knn.pkl'),
        ('rf_model_path', 'model/model_rf.pkl', 'model/rf/model_rf.pkl'),
        ('dt_model_path', 'model/model_dt.pkl', 'model/dt/model_dt.pkl'),
        ('lr_model_path', 'model/model_lr.pkl', 'model/lr/model_lr.pkl'),
        ('textcnn_vocab_path', 'model/textcnn_vocab.json', 'model/textcnn/textcnn_vocab.json'),
        ('textcnn_weight_path', 'model/textcnn.pt', 'model/textcnn/textcnn.pt'),
        ('textrcnn_vocab_path', 'model/textcnn_vocab.json', 'model/textrcnn/textcnn_vocab.json'),
        ('textrcnn_weight_path', 'model/textrcnn.pt', 'model/textrcnn/textrcnn.pt'),
    ]
    for key, old_v, new_v in cfg_rows:
        SystemConfig.objects.filter(key=key, value=old_v).update(value=new_v)

    file_rows = [
        ('model/model_svm.pkl', 'model/svm/model_svm.pkl'),
        ('model/model_knn.pkl', 'model/knn/model_knn.pkl'),
        ('model/model_rf.pkl', 'model/rf/model_rf.pkl'),
        ('model/model_dt.pkl', 'model/dt/model_dt.pkl'),
        ('model/model_lr.pkl', 'model/lr/model_lr.pkl'),
        ('model/textcnn.pt', 'model/textcnn/textcnn.pt'),
        ('model/textrcnn.pt', 'model/textrcnn/textrcnn.pt'),
    ]
    for old_f, new_f in file_rows:
        ModelInfo.objects.filter(file_path=old_f).update(file_path=new_f)

    ModelInfo.objects.filter(
        model_type='textrcnn', vectorizer_path='model/textcnn_vocab.json'
    ).update(vectorizer_path='model/textrcnn/textcnn_vocab.json')

    ModelInfo.objects.filter(vectorizer_path='model/textcnn_vocab.json').update(
        vectorizer_path='model/textcnn/textcnn_vocab.json'
    )

    ModelInfo.objects.filter(vectorizer_path='model/tfidfVectorizer.pkl').update(
        vectorizer_path='model/svm/tfidfVectorizer.pkl'
    )


def _noop(apps, schema_editor):
    pass


class Migration(migrations.Migration):
    dependencies = [
        ('app', '0015_fusionconfigpreset'),
    ]

    operations = [
        migrations.RunPython(_migrate, _noop),
    ]
