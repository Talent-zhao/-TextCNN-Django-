---
language: en
license: mit
tags:
  - text-classification
  - depression
  - mental-health
  - huggingface
datasets:
  - thePixel42/depression-detection
  - infamouscoder/depression-reddit-cleaned
model-index:
  - name: DistilBERT for Depression Detection
    results:
      - task:
          name: Text Classification
          type: text-classification
        metrics:
          - name: Evaluation Loss
            type: loss
            value: 0.0631
---

# DistilBERT for Depression Detection

This model is a fine-tuned version of `distilbert-base-uncased` for binary depression classification based on Reddit and mental health-related posts.

## 📊 Training Details

- **Base model**: distilbert-base-uncased
- **Epochs**: 3
- **Batch size**: 8 (train), 16 (eval)
- **Optimizer**: AdamW with weight decay
- **Loss function**: CrossEntropyLoss
- **Hardware**: Trained using GPU acceleration

## 🧾 Datasets Used

- [thePixel42/depression-detection](https://huggingface.co/datasets/thePixel42/depression-detection)
- [infamouscoder/depression-reddit-cleaned](https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned)

The datasets were cleaned to remove rows with missing `text`, labels were binarized (0 = not depressed, 1 = depressed), and duplicates were removed.

## 🧪 Evaluation

| Metric              | Value     |
|---------------------|-----------|
| Loss                | 0.0631    |
| Samples/sec         | 85.56     |
| Steps/sec           | 5.35      |

## 🚀 Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained("your-username/depression-detection-model")
tokenizer = AutoTokenizer.from_pretrained("your-username/depression-detection-model")

inputs = tokenizer("I feel sad and hopeless", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
    predicted_class = torch.argmax(logits).item()

print("Prediction:", predicted_class)
