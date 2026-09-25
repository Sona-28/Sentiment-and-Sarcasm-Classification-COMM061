# BESSTIE Sentiment & Sarcasm Classification Across English Varieties

A natural language processing project investigating **sentiment and sarcasm classification across Australian English (en-AU), British English (en-UK), and Indian English (en-IN)** using classical machine-learning baselines, pretrained Transformers, cross-variety evaluation, LoRA-adapted LLMs, interpretability, error analysis, and model deployment.

## Overview

The project uses the **BESSTIE dataset**, containing **6,243 text samples** across three English varieties:

| English variety | Samples |
|---|---:|
| en-IN | 2,332 |
| en-UK | 2,004 |
| en-AU | 1,907 |
| **Total** | **6,243** |

Two binary classification tasks are investigated:

- **Sentiment:** positive vs. negative
- **Sarcasm:** sarcastic vs. non-sarcastic

The project focuses particularly on whether models trained on one English variety generalise to other varieties and how linguistic variation affects sarcasm detection.

## Key Findings

### Dataset analysis

- Sentiment is relatively balanced overall: **51% negative and 49% positive**.
- Sarcasm is highly imbalanced, with only **13.98%** of samples labelled sarcastic.
- Sarcasm is substantially more prevalent in **en-AU (29.42%)** than in en-IN (6.82%) and en-UK (7.63%).
- Sarcasm is concentrated more heavily in Reddit data than Google reviews.
- Question marks occur considerably more often in sarcastic text than non-sarcastic text in the analysed dataset.
- **28.11%** of texts contain at least one non-English word, with the highest proportion occurring in en-IN.

### Vocabulary similarity

Vocabulary overlap and distributional similarity were analysed using **Jaccard similarity** and **TF-IDF cosine similarity**.

The training-data analysis reported:

| Comparison | Jaccard | Cosine |
|---|---:|---:|
| en-UK vs en-AU | 0.3461 | 0.9877 |
| en-UK vs en-IN | 0.2928 | 0.9497 |
| en-AU vs en-IN | 0.2867 | 0.9557 |

These results indicate substantial distributional similarity across varieties while also showing lower lexical overlap involving Indian English.

## Models

### Classical baselines

The project evaluates:

- TF-IDF + Logistic Regression
- TF-IDF + Linear SVC
- TF-IDF + Multinomial Naive Bayes

The baseline pipeline includes text normalisation, spaCy tokenisation, removal of unwanted tokens, contraction expansion, custom stopword removal, POS filtering, lemmatisation, and removal of very short tokens.

### Pretrained Transformers

The following pretrained models were fine-tuned:

- **BERT**
- **RoBERTa**
- **DistilBERT**

Hyperparameters such as learning rate and number of epochs were tuned using validation data.

### Cross-variety evaluation

**RoBERTa-base** was used for the cross-variety sentiment experiment. Separate models were trained on en-AU, en-UK, and en-IN and evaluated across all three test varieties.

The reported Macro-F1 matrix was:

| Train / Test | en-AU | en-UK | en-IN |
|---|---:|---:|---:|
| en-AU | 0.887 | 0.940 | 0.825 |
| en-UK | 0.850 | 0.946 | 0.847 |
| en-IN | 0.889 | 0.936 | 0.843 |

### Qwen + LoRA

For variety-specific sarcasm detection, the project uses **Qwen3-1.7B** with lightweight **LoRA adapters**.

The adapters use:

- LoRA rank: **16**
- LoRA alpha: **32**
- LoRA dropout: **0.05**
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Maximum sequence length: **384**
- Optimiser: **AdamW**
- Weight decay: **0.01**
- Warmup ratio: **0.1**
- Seeds: **42** and **52**

Separate adapters were trained for en-AU, en-UK, and en-IN while keeping the Qwen base model frozen.

Class-weighted loss was used to address sarcasm imbalance, and the classification threshold was tuned on the validation set over the range **0.20–0.80**.

## Evaluation

The primary evaluation metric is **Macro-F1**, supported by:

- Macro-precision
- Macro-recall
- Class-specific precision
- Class-specific recall
- Class-specific F1
- Accuracy
- Confusion matrices

Macro-F1 was prioritised because the sarcasm task is strongly class-imbalanced.

### Baseline vs Transformer results

Reported results include:

| Task | Model | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Sentiment | TF-IDF + Logistic Regression | 0.834 | 0.833 |
| Sentiment | RoBERTa | 0.898 | 0.898 |
| Sentiment | BERT | 0.888 | — |
| Sentiment | DistilBERT | 0.878 | — |
| Sarcasm | TF-IDF + Logistic Regression | 0.745 | 0.624 |
| Sarcasm | BERT | 0.864 | 0.662 |
| Sarcasm | DistilBERT | — | 0.638 |
| Sarcasm | RoBERTa | — | 0.581 |

The experiments show a larger performance improvement from contextual Transformers for sentiment than for sarcasm. Sarcasm remains difficult because it often depends on implicit context, intent, tone, pragmatic cues, and cultural or variety-specific expressions.

## LoRA Cross-Variety Results

The LoRA-based sarcasm experiment reported approximately:

- en-AU in-domain Macro-F1: **0.74**
- en-UK in-domain Macro-F1: **0.70**
- en-IN in-domain Macro-F1: **0.61**

Cross-variety performance decreased substantially in several settings. For example:

- en-AU → en-IN: approximately **0.52 Macro-F1**
- en-IN → en-AU: approximately **0.42 Macro-F1**

Across the evaluated combinations, non-sarcasm F1 ranged from approximately **0.807 to 0.962**, while sarcasm F1 was considerably more variable, ranging from approximately **0.020 to 0.649**.

These results indicate that lightweight adaptation improves variety-specific modelling efficiency but does not by itself eliminate cross-variety generalisation difficulties.

## Sarcasm Error Analysis & Few-Shot Prompting

The project also analyses errors produced by a LoRA-based sarcasm classifier.

### Error analysis procedure

1. Extract 10 misclassified examples.
2. Keep the sample balanced:
   - 5 sarcastic
   - 5 non-sarcastic
3. Select 4 representative examples for few-shot prompting:
   - 2 sarcastic
   - 2 non-sarcastic
4. Evaluate three prompt variants on the remaining 6 examples.

### Prompt variants

- **v1 — Baseline Few-Shot:** labelled examples with explanations.
- **v2 — Reasoning Prompt:** adds step-by-step reasoning instructions.
- **v3 — Constrained Prompt:** adds stricter rules and a bias toward non-sarcastic predictions.

Reported results:

| Prompt | Correct predictions | Accuracy |
|---|---:|---:|
| v1 — Baseline Few-Shot | 5/6 | 83.3% |
| v2 — Reasoning | 3/6 | 50.0% |
| v3 — Constrained | 2/6 | 33.3% |

The experiment suggests that prompt construction strongly affects sarcasm predictions. The simpler few-shot prompt produced the highest accuracy on this small evaluation sample, while the reasoning prompt over-predicted sarcasm and the constrained prompt became more conservative.

## Interpretability

**LIME** and **SHAP** were used to analyse the RoBERTa sentiment classifier.

The analysis examined which words contributed to positive and negative predictions and indicated that the classifier relied substantially on lexical sentiment cues. This helped investigate why sentiment transferred relatively well between some English varieties while performance changed for Indian English.

## Deployment

The final system is designed as an interactive web application using:

- **Gradio** for the frontend
- **Python** for backend inference
- **Hugging Face Spaces** for deployment
- **Hugging Face Hub** for model storage

### Deployment architecture

```text
User
 │
 ▼
Gradio Interface
 │
 ├── Task: Sentiment
 │      └── Select English variety
 │             └── Variety-specific RoBERTa model
 │
 └── Task: Sarcasm
        └── Select English variety
               └── Shared Qwen3-1.7B
                      └── Variety-specific LoRA adapter
 │
 ▼
Prediction
```

Models are loaded once when the application starts to avoid repeated model initialisation and downloads.

For sarcasm detection, a shared Qwen base model is combined with variety-specific LoRA adapters. The active adapter can be switched without loading a separate full 1.7B-parameter model for every variety.

## Inference Efficiency

The report compares TF-IDF + Logistic Regression and RoBERTa for sentiment inference.

Selected reported measurements include:

| Scenario | Logistic Regression | RoBERTa |
|---|---:|---:|
| Single input, CPU | ~0.24 ms | ~384 ms |
| Single input, GPU | ~0.64 ms | ~15 ms |
| Stress test, CPU | ~3,669 req/s | ~13 req/s |
| Stress test, GPU | ~3,171 req/s | ~115 req/s |

The efficiency experiments demonstrate the trade-off between lightweight classical models and more computationally expensive Transformer models.

## Technology Stack

- Python
- pandas
- NumPy
- spaCy
- Wordfreq
- scikit-learn
- PyTorch
- Hugging Face Transformers
- PEFT / LoRA
- BERT
- RoBERTa
- DistilBERT
- Qwen3-1.7B
- LIME
- SHAP
- Gradio
- Hugging Face Spaces
- Hugging Face Hub

## Project Workflow

```text
BESSTIE Dataset
      │
      ▼
Exploratory Data Analysis
      │
      ├── Sentiment distribution
      ├── Sarcasm distribution
      ├── English-variety analysis
      ├── Vocabulary analysis
      └── Linguistic features
      │
      ▼
Text Preprocessing
      │
      ▼
Baseline Models
      │
      ├── Logistic Regression
      ├── Linear SVC
      └── Multinomial Naive Bayes
      │
      ▼
Transformer Fine-Tuning
      │
      ├── BERT
      ├── RoBERTa
      └── DistilBERT
      │
      ▼
Cross-Variety Evaluation
      │
      ▼
Qwen3-1.7B + LoRA
      │
      ▼
Sarcasm Error Analysis
      │
      ▼
Few-Shot Prompting
      │
      ▼
Model Deployment
      │
      ▼
Gradio + Hugging Face Spaces
```

## Reproducibility

For reproducible experiments:

1. Keep the original train, validation, and test splits unchanged.
2. Use the documented preprocessing pipeline.
3. Use the specified random seeds where applicable.
4. Perform model selection using validation data.
5. Keep the test set reserved for final evaluation.
6. For sarcasm classification, perform threshold tuning only on the validation set.
7. Report Macro-F1 alongside per-class metrics because of the class imbalance.

## Limitations

The experiments highlight several limitations:

- Sarcasm is strongly underrepresented in the dataset.
- Sarcasm distribution differs substantially across English varieties.
- Domain differences between Reddit and Google reviews may introduce additional bias.
- Cross-variety sarcasm performance can deteriorate considerably.
- LoRA reduces the number of trainable parameters but does not inherently solve linguistic transfer.
- The few-shot prompting evaluation used only a small set of six remaining error samples, so its results should not be treated as a general performance estimate.
- Transformer inference is substantially more computationally expensive than the TF-IDF baseline.
