---
language:
- en
tags:
- mental health
- sentiment analysis
- classification
- bert
- transformers
license: apache-2.0
datasets:
- nikhileswarkomati/suicide-watch
- suchintikasarkar/sentiment-analysis-for-mental-health
- custom/reddit-mental-health
- custom/mental-health-social-media-posts
- ourafla/Mental-Health_Text-Classification_Dataset
metrics:
- accuracy
- f1
base_model: mental/mental-bert-base-uncased
pipeline_tag: text-classification
---


# MentalHealthBERT

A fine-tuned BERT-based model for multi-class mental health text classification, achieving **89.7% accuracy** on held-out test data.

---

## Model Description

This model is a fine-tuned version of [mental/mental-bert-base-uncased](https://huggingface.co/mental/mental-bert-base-uncased) designed to classify text into four mental health categories:
- **Anxiety**
- **Depression** 
- **Normal**
- **Suicidal**

**Base Model:** MentalBERT (BERT-Base uncased, pre-trained on mental health-related Reddit posts)
**Architecture:** BertForSequenceClassification with 4 output labels  
**Parameters:** ~110M (BERT-Base: 12 layers, 768 hidden dimensions, 12 attention heads)

---

## Intended Use

### Primary Use Cases
- Research on mental health text analysis
- Early detection support systems for mental health concerns in online social content
- Sentiment analysis in mental health contexts
- Supporting mental health monitoring and research

### Target Users
- Mental health researchers
- Clinical researchers
- Data scientists working on mental health NLP projects
- Social workers and support organizations

**⚠️ Important:** This model is **NOT** intended for clinical diagnosis. It is a supplementary research tool and should not replace professional mental health evaluation or therapy. Model predictions are not psychiatric diagnoses, and anyone struggling with mental health issues should seek professional help.

---

## Training Data


### Datasets

The model was trained on a combined dataset from multiple sources, which were then integrated into a unified 4‑class corpus published as `ourafla/Mental-Health_Text-Classification_Dataset`.

1. **Mental Health Text Classification Dataset (4‑Class)**  
   - Curated, cleaned, and relabeled 4‑class dataset combining several public mental‑health corpora  
   - Hosted on Hugging Face Hub as:  
     `ourafla/Mental-Health_Text-Classification_Dataset`  
   - Reference:  
     > Mukherjee, P. (2025). *Mental Health Text Classification Dataset (4‑Class)* [Dataset]. Hugging Face Hub. https://huggingface.co/datasets/ourafla/Mental-Health_Text-Classification_Dataset  

2. **Suicide Detection Dataset**  
   - 232,074 samples  
   - [Kaggle link](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)

3. **Mental Health Social Media Posts**  
   - Reddit-based posts (downloaded from this folder)  
   - [Google Drive link](https://drive.google.com/drive/folders/11aW_fpXjA-O51uv3xYY3xj6NWGh1VYh_)

4. **Sentiment Analysis for Mental Health**  
   - Over 44,000 samples  
   - [Kaggle link](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)

The final training corpus used for this model corresponds to the processed version released as `ourafla/Mental-Health_Text-Classification_Dataset`.

### Data Preprocessing
- Text normalization and cleaning
- Label standardization across datasets
- Duplicate removal
- Class balancing to ensure equal representation (248 samples per class in test set)

### Data Split
- **Training:** ~49,382 samples (balanced across 4 classes)
- **Validation:** ~5,487 samples (10% holdout)
- **Test:** 992 samples (248 per class, balanced)

---

## Training Procedure

### Hyperparameters
- **Optimizer:** AdamW
  - Learning rate: 2e-5
  - Weight decay: 1e-2
- **Epochs:** 5
- **Batch size:** 16 (training), 32 (validation/test)
- **Max sequence length:** 128 tokens
- **Scheduler:** Linear warmup (10% of training steps)
- **Loss function:** CrossEntropyLoss with class weights

### Training Environment
- **Platform:** Google Colab with GPU (Tesla T4)
- **Framework:** PyTorch with Hugging Face Transformers 4.45.1
- **Training time:** ~80.39 minutes (5 epochs)

### Training Phases
The model underwent 3 development phases:
1. **Phase 1:** Initial baseline training
2. **Phase 2:** Refined preprocessing and model optimization
3. **Phase 3:** Enhanced training with improved data balancing and class weights

---

## Performance

### Test Set Results (Phase 3)

| Metric | Score |
|--------|-------|
| **Accuracy** | 89.72% |
| **Macro Precision** | 89.56% |
| **Macro Recall** | 89.72% |
| **Macro F1-Score** | 89.54% |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Anxiety** | 0.88 | 0.85 | 0.87 | 248 |
| **Depression** | 0.86 | 0.78 | 0.82 | 248 |
| **Normal** | 0.94 | 0.98 | 0.96 | 248 |
| **Suicidal** | 0.91 | 0.98 | 0.94 | 248 |

### Key Observations
- **Strongest performance:** Normal (96% F1) and Suicidal (94% F1) classes
- **Moderate performance:** Anxiety (87% F1) and Depression (82% F1) classes
- **Challenge:** Some confusion between Anxiety and Depression classes (common in mental health classification)
- The model demonstrates strong generalization across all four mental health categories

---

## 🔍 Advanced Model Evaluation & Error Analysis

To better understand the model’s behaviour beyond aggregate metrics, an additional evaluation notebook is provided on Kaggle. This analysis focuses on class-wise errors, confusion patterns, and probability calibration, with particular attention to uncertainty in linguistically overlapping categories such as Anxiety and Depression.

The intent of this evaluation is not to claim clinical reliability, but to transparently examine where the model performs well and where it remains limited.

* **Kaggle Evaluation Notebook:**
  [https://www.kaggle.com/code/priyangshumukherjee/mental-health-bert-fine-tuned-evaluation](https://www.kaggle.com/code/priyangshumukherjee/mental-health-bert-fine-tuned-evaluation)

---

## Usage

### Installation
```bash
pip install transformers torch
```

### Basic Inference
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
model_name = "mental/mental-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

# Load fine-tuned weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("best_phase3.pth", map_location=device))
model.to(device)
model.eval()

# Define label mapping
id2label = {0: "Anxiety", 1: "Depression", 2: "Normal", 3: "Suicidal"}

# Example inference
text = "I've been feeling really overwhelmed and anxious lately"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    
print(f"Predicted class: {id2label[prediction]}")
```

### Batch Inference
```python
texts = [
    "I feel hopeless and don't see the point anymore",
    "Had a great day today, feeling positive!",
    "My heart is racing and I can't stop worrying"
]

inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    
for text, pred in zip(texts, predictions):
    print(f"Text: {text[:50]}...")
    print(f"Predicted: {id2label[pred.item()]}\n")
```

---

## Limitations and Considerations

### Known Limitations
1. **Not a diagnostic tool:** Cannot replace professional mental health assessment
2. **Text-only analysis:** Does not consider non-verbal cues, medical history, or clinical context
3. **Class imbalance challenges:** Some confusion between Depression and Anxiety categories
4. **Language bias:** Trained primarily on English text from social media
5. **Cultural context:** May not generalize well across different cultural expressions of mental health
6. **Temporal limitations:** Trained on historical data; language use evolves

### Ethical Considerations
- **Privacy:** All training data was from publicly available, anonymized sources
- **Bias:** Model may reflect biases present in training data (Reddit demographics)
- **Responsible use:** Should be used as a screening tool only, not for definitive diagnosis
- **Professional oversight:** Any clinical applications must involve mental health professionals
- **Informed consent:** Users should be aware that their text is being analyzed

### Potential Biases
- Reddit user demographics (younger, predominantly Western)
- Self-reported mental health states (not clinically verified)
- Language and expression styles specific to online communities
- Underrepresentation of certain mental health conditions

---

## Model Card Authors

Priyangshu Mukherjee

---

## Model Card Contact

For questions or concerns about this model:
- Email: priyangshumukherjeebtech24@rvu.edu.in
- Issues: [Create an issue in the model repository]

---

## Citation

If you use this model in your research, please cite:

```bibtex
@software{mental_health_classifier_2025,
  author = {Mukherjee, Priyangshu},
  title = {Mental Health Text Classifier (MentalBERT Fine-tuned)},
  year = {2025},
  note = {Fine-tuned model for multi-class mental health text classification}
}
```

**Base Model Citation:**
```bibtex
@inproceedings{ji2022mentalbert,
  title = {{MentalBERT: Publicly Available Pretrained Language Models for Mental Healthcare}},
  author = {Shaoxiong Ji and Tianlin Zhang and Luna Ansari and Jie Fu and Prayag Tiwari and Erik Cambria},
  year = {2022},
  booktitle = {Proceedings of LREC}
}
```

---

## Acknowledgments

- **Base Model:** mental/mental-bert-base-uncased by Shaoxiong Ji et al.
- **Frameworks:** Hugging Face Transformers, PyTorch
- **Datasets:** Suicide Detection Dataset, Reddit Mental Health Posts
- **Compute:** Google Colab GPU resources

---

## License

This model is released for research and non-commercial use. Please check the base model license at [mental/mental-bert-base-uncased](https://huggingface.co/mental/mental-bert-base-uncased) for additional terms.

---

## Additional Resources

- [MentalBERT Paper (LREC 2022)](https://aclanthology.org/2022.lrec-1.322/)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [Mental Health Research Guidelines](https://www.nimh.nih.gov/)
- [Crisis Resources: National Suicide Prevention Lifeline (US): 988]

---

## Version History

- **v3.0 (Phase 3):** Enhanced model with improved data balancing and class weights
  - Test Accuracy: 89.72%
  - Macro F1: 89.54%
  - Balanced 4-class classification
  
- **v2.0 (Phase 2):** Refined preprocessing and training procedures
  
- **v1.0 (Phase 1):** Initial baseline model

---

## Technical Specifications

### Model Architecture
```
BertForSequenceClassification(
  (bert): BertModel(
    (embeddings): BertEmbeddings(...)
    (encoder): BertEncoder(
      12 x BertLayer(...)
    )
    (pooler): BertPooler(...)
  )
  (dropout): Dropout(p=0.1)
  (classifier): Linear(768 -> 4)
)
```

### Tokenization
- **Tokenizer:** BertTokenizer (uncased)
- **Vocabulary size:** 30,522
- **Special tokens:** [CLS], [SEP], [PAD], [UNK], [MASK]
- **Max position embeddings:** 512 (used: 128)

### Input Format
- **Text preprocessing:** Lowercase, Unicode normalization
- **Padding:** Max length (128 tokens)
- **Truncation:** Enabled
- **Return type:** PyTorch tensors

---