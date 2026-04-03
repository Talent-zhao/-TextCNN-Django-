Best_Modle — 外部抑郁/心理健康文本分类模型（本地下载）
====================================================

本目录存放从 Hugging Face 拉取的 Transformers 序列分类权重，供研究或二次集成使用。

与项目内置预测融合（model/svm、model/textcnn 等）不是同一格式：后台「预测融合」直接使用的是
.pkl + TF-IDF 或 .pt + 词表；若要使用本目录模型，需在代码中通过 transformers + torch 调用，
或自行导出为项目支持的格式。

--------------------------------------------------------------------
1) 二分类（抑郁相关 vs 非抑郁）
--------------------------------------------------------------------
目录：binary_depression_distilbert/
来源：https://huggingface.co/TRT1000/depression-detection-model
许可证：以模型卡为准（卡片声明 MIT）
架构：DistilBERT + 序列分类头
数据：英文 Reddit / 心理健康相关帖子风格；标签约为 0=非抑郁，1=抑郁（见该目录 README.md）

--------------------------------------------------------------------
2) 四分类（心理健康多类，含 Depression）
--------------------------------------------------------------------
目录：four_class_mental_health_bert/
来源：https://huggingface.co/ourafla/mental-health-bert-finetuned
许可证：Apache-2.0（见模型卡）
架构：BERT-base + 4 类分类；英文
类别（与训练说明一致，顺序以模型 logits 索引为准，一般为）：
  0 — Anxiety
  1 — Depression
  2 — Normal
  3 — Suicidal

⚠ 非临床诊断工具，仅供研究与教学；不可用于正式筛查/诊断。

--------------------------------------------------------------------
国内下载镜像（若直连 huggingface.co 超时）
--------------------------------------------------------------------
PowerShell 示例：
  $env:HF_ENDPOINT="https://hf-mirror.com"
  pip install huggingface_hub
  python Best_Modle\download_hf_models.py

或自行使用：
  huggingface-cli download TRT1000/depression-detection-model --local-dir ...
  huggingface-cli download ourafla/mental-health-bert-finetuned --local-dir ...

--------------------------------------------------------------------
最小推理示例（需单独安装：pip install transformers torch）
--------------------------------------------------------------------
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def predict_local(model_dir, texts):
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
    mdl.eval()
    out = []
    for t in texts if isinstance(texts, list) else [texts]:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = mdl(**enc).logits
            pred = int(torch.argmax(logits, dim=-1).item())
        out.append(pred)
    return out

# 二分类示例（路径按本项目相对根目录调整）
# predict_local("Best_Modle/binary_depression_distilbert", "I feel hopeless")

# 四分类 id→语义参见上文类别列表
