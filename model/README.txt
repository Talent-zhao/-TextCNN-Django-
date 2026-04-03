模型目录约定（相对本项目根目录）
================================

model/
  svm/          SVM：model_svm.pkl、tfidfVectorizer.pkl、tfidfVectorizer_svm.pkl
  knn/          KNN 及对应向量器
  rf/           随机森林
  dt/           决策树
  lr/           逻辑回归
  textcnn/      textcnn.pt、textcnn_vocab.json
  textrcnn/     textrcnn.pt、textcnn_vocab.json（词表文件名与 TextCNN 侧一致）
  num2name.json                    类别编号→标签（各脚本统一写此文件，供融合默认引用）
  model_training_registry.json     训练次数、召回等指标登记

训练脚本默认将产物写入对应子目录；旧版扁平路径仍可在预测服务中自动回退（建议迁移后执行
python scripts/migrate_model_flat_layout.py 并运行 Django 迁移 0016）。
