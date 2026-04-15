# Results Analysis — Paraphrase Detection & Semantic Textual Similarity

## 1. Project Overview

This project evaluates deep learning architectures for two core NLP tasks: **Paraphrase Detection** (binary classification) and **Semantic Textual Similarity** (regression). The goal is to compare traditional neural architectures using static embeddings against modern transformer-based models with contextualised representations.

### Models Evaluated

| Category | Models | Embedding Type | Key Characteristic |
|----------|--------|----------------|--------------------|
| **Siamese Baselines** | Siamese-LSTM, Siamese-GRU | Static GloVe 100d (frozen) | Lightweight, fast inference |
| **Fine-tuned Transformers** | BERT, RoBERTa, DistilBERT | Contextualised (all params trained) | Large, high accuracy |
| **Sentence Transformer** | SBERT | Pre-trained sentence embeddings | Zero-shot, no fine-tuning needed |

### Datasets

| Dataset | Task | Size (Train / Val) | Metric |
|---------|------|-------------------|--------|
| **MRPC** (Microsoft Research Paraphrase Corpus) | Binary Classification | 3,668 / 408 | Accuracy, F1 |
| **STS-B** (Semantic Textual Similarity Benchmark) | Regression (0–5 scale) | 5,749 / 1,500 | Pearson r, Spearman ρ |
| **QQP** (Quora Question Pairs) | Binary Classification | 10,000 / 2,000 (subset) | Accuracy, F1 |

---

## 2. MRPC Results — Paraphrase Detection

MRPC tests whether two sentences drawn from news articles are semantic paraphrases. It is a small but challenging dataset with ~67% positive class imbalance, making **F1 score** the more reliable metric.

### 2.1 Classification Performance

| Model | Accuracy | F1 Score | Acc. Rank | F1 Rank |
|-------|:--------:|:--------:|:---------:|:-------:|
| Siamese-LSTM | 72.06% | 0.8167 | 4th | 5th |
| Siamese-GRU | 69.61% | 0.8171 | 5th | 4th |
| BERT | 84.56% | 0.8901 | 3rd | 3rd |
| DistilBERT | 86.76% | 0.9082 | 2nd | 2nd |
| **RoBERTa** | **89.22%** | **0.9206** | **1st** | **1st** |

**Key Findings:**

- **RoBERTa dominates** with 89.22% accuracy and 0.9206 F1 — a clear margin above all other models. Its advantage comes from an optimised pre-training recipe: dynamic masking, larger batch sizes, more training data (160GB vs BERT's 16GB), and removal of the Next Sentence Prediction objective.
- **DistilBERT outperforms BERT** (86.76% vs 84.56% accuracy) despite having only 61% of BERT's parameters. This suggests that with only 2 training epochs, the lighter model generalises better by avoiding overfitting on MRPC's small training set.
- **Siamese-LSTM slightly outperforms Siamese-GRU in accuracy** (72.06% vs 69.61%) but **GRU has marginally better F1** (0.8171 vs 0.8167). The near-identical F1 scores indicate both models learn similar decision boundaries with static embeddings — the 2.4% accuracy gap likely reflects variance on the small validation set.
- The **~17 percentage point accuracy gap** between the best Siamese model and RoBERTa confirms that contextualised representations are fundamentally superior for paraphrase detection.

![MRPC Accuracy Comparison](plots/mrpc/accuracy.png)

![MRPC F1 Score Comparison](plots/mrpc/f1.png)

### 2.2 Inference Time

| Model | Time (s) | Relative to Fastest |
|-------|:--------:|:-------------------:|
| **Siamese-LSTM** | **0.112** | **1.0×** |
| Siamese-GRU | 0.229 | 2.0× |
| DistilBERT | 1.430 | 12.8× |
| BERT | 2.460 | 22.0× |
| RoBERTa | 2.721 | 24.3× |

- Siamese-LSTM is **24× faster** than RoBERTa. For latency-sensitive applications where moderate accuracy is acceptable, this is a significant advantage.
- **DistilBERT is 1.9× faster than BERT** while delivering better accuracy — making it the best choice when balancing speed and quality.
- BERT and RoBERTa have similar inference times (~2.5–2.7s) as expected from their identical 12-layer architectures.

![MRPC Time Comparison](plots/mrpc/time.png)

### 2.3 Model Complexity Trade-off

| Model | Total Params | Trainable Params | Size (Millions) |
|-------|:------------:|:----------------:|:---------------:|
| Siamese-GRU | 1,491,046 | 250,946 | 1.49M |
| Siamese-LSTM | 1,549,926 | 309,826 | 1.55M |
| DistilBERT | 66,955,010 | 66,955,010 | 66.96M |
| BERT | 109,483,778 | 109,483,778 | 109.48M |
| RoBERTa | 124,647,170 | 124,647,170 | 124.65M |

- Transformer models are **43–84× larger** than Siamese baselines.
- Siamese models freeze GloVe embeddings — only ~20% of their parameters (the recurrent + FC layers) are trainable. This limits their capacity but drastically reduces memory and compute requirements.
- **DistilBERT sits at the efficiency sweet spot**: at 67M params, it achieves 86.76% accuracy — only 2.5 points below RoBERTa's 125M params.

![MRPC Complexity Trade-off](plots/mrpc/complexity_tradeoff.png)

The scatter plot shows a clear **diminishing returns** pattern: the jump from 1.5M to 67M parameters yields a massive accuracy gain (~15 points), but doubling again to 125M only adds ~2.5 points more.

---

## 3. STS-B Results — Semantic Textual Similarity

STS-B is a **regression task** where each sentence pair receives a continuous similarity score from 0.0 (unrelated) to 5.0 (semantically equivalent). Models are evaluated on how well their predicted scores correlate with human judgments.

### 3.1 Correlation Performance

| Model | Pearson r | Spearman ρ | Pearson Rank |
|-------|:---------:|:----------:|:------------:|
| Siamese-LSTM | 0.1769 | 0.1709 | 6th |
| Siamese-GRU | 0.6039 | 0.5996 | 5th |
| DistilBERT | 0.8625 | 0.8600 | 4th |
| SBERT (zero-shot) | 0.8696 | 0.8672 | 3rd |
| BERT | 0.8843 | 0.8809 | 2nd |
| **RoBERTa** | **0.8863** | **0.8849** | **1st** |

**Key Findings:**

- **RoBERTa again leads**, achieving Pearson r = 0.886 and Spearman ρ = 0.885, indicating its predicted similarity scores have a near-linear relationship with human judgments.
- **BERT is a close second** (Pearson = 0.884), only 0.002 behind RoBERTa. Unlike on MRPC where DistilBERT outperformed BERT, here the larger model's capacity provides a slight edge on the more nuanced regression task.
- **SBERT performs remarkably well without any fine-tuning** (Pearson = 0.870). As a zero-shot model using pre-computed sentence embeddings and cosine similarity, it outperforms DistilBERT (0.863) — which actually underwent supervised fine-tuning on STS-B. This highlights the power of sentence-level pre-training objectives.
- **Siamese-LSTM essentially fails** on STS-B with Pearson = 0.177, barely above random correlation. Static GloVe embeddings coupled with an LSTM encoder are insufficient to produce meaningful graded similarity scores.
- **Siamese-GRU performs much better** (Pearson = 0.604) than LSTM (0.177) — a striking 0.43 correlation gap. The GRU may be learning better compositional representations with its simpler gating mechanism, or the LSTM is suffering from training instability on this regression task.
- **Pearson and Spearman correlations are extremely close** for all models, indicating that the predicted-vs-true similarity relationship is approximately linear (not just monotonic).

![STS-B Correlation Comparison](plots/sts/sts_correlations.png)

### 3.2 The SBERT Anomaly

SBERT (Sentence-BERT) achieves **0.870 Pearson correlation with zero fine-tuning** — it simply encodes each sentence with a pre-trained sentence transformer and computes cosine similarity. This outperforms fine-tuned DistilBERT (0.863) and approaches BERT (0.884).

This result has important practical implications:
- For STS tasks, SBERT may be a **better default choice** than fine-tuning smaller transformers, especially when labelled data is limited.
- SBERT's sentence-level pre-training (using NLI data with contrastive objectives) appears to produce representations that naturally encode graded similarity without task-specific supervision.

---

## 4. QQP Results — Quora Question Pairs

QQP tests whether two questions asked on Quora are semantically equivalent. It is much larger than MRPC (~364K pairs total), though we use a 10K/2K subset for tractable training.

### 4.1 Classification Performance

| Model | Accuracy | F1 Score | Acc. Rank |
|-------|:--------:|:--------:|:---------:|
| Siamese-LSTM | 67.60% | 0.6241 | 4th |
| Siamese-GRU | 64.40% | 0.6364 | 5th |
| BERT | 81.20% | 0.7656 | 3rd |
| DistilBERT | 81.95% | 0.7519 | 2nd |
| **RoBERTa** | **82.05%** | **0.7726** | **1st** |

**Key Findings:**

- **RoBERTa is the best model again** (82.05% accuracy, 0.773 F1), but the transformer models are **much more tightly clustered** on QQP than on MRPC — only 0.85 percentage points separate BERT from RoBERTa.
- **All models score lower on QQP than MRPC.** This is expected for several reasons:
  - QQP's question-question pairs require understanding of *interrogative intent*, not just surface meaning
  - We only trained on a 10K subset (vs MRPC's full 3.7K), giving less data per sample complexity
  - QQP has more balanced class distribution (~37% positive), so the class imbalance boost that inflated MRPC F1 scores doesn't apply
- **Siamese-GRU has better F1 but worse accuracy** than LSTM on QQP (identical pattern to MRPC). The GRU's slightly higher recall compensates for lower precision.
- The **accuracy gap narrows** on QQP: Siamese models are ~14–18 points below transformers (vs ~17–20 on MRPC), possibly because the subset training limits transformer models' data advantage.

![QQP Accuracy Comparison](plots/qqp/accuracy.png)

![QQP F1 Score Comparison](plots/qqp/f1.png)

### 4.2 Inference Time

| Model | Time (s) | Relative to Fastest |
|-------|:--------:|:-------------------:|
| **Siamese-LSTM** | **0.522** | **1.0×** |
| Siamese-GRU | 1.140 | 2.2× |
| DistilBERT | 6.780 | 13.0× |
| RoBERTa | 12.935 | 24.8× |
| BERT | 13.367 | 25.6× |

On the larger QQP validation set (2,000 samples vs MRPC's 408), inference times scale proportionally. The relative speed ratios remain consistent with MRPC, confirming that the Siamese speed advantage is structural, not dataset-dependent.

![QQP Time Comparison](plots/qqp/time.png)

### 4.3 Model Complexity Trade-off

![QQP Complexity Trade-off](plots/qqp/complexity_tradeoff.png)

On QQP, the transformer models cluster much more tightly in both accuracy and F1, while the gap from Siamese baselines remains large. This confirms that the Siamese-to-transformer jump represents a qualitative capability difference (static → contextualised), while differences *among* transformers are refinements.

---

## 5. Cross-Dataset Comparison

### 5.1 Best Model Performance Across Datasets

| Dataset | Task | Best Model | Score |
|---------|------|-----------|-------|
| MRPC | Classification | RoBERTa | 89.22% acc / 0.921 F1 |
| STS-B | Regression | RoBERTa | 0.886 Pearson / 0.885 Spearman |
| QQP | Classification | RoBERTa | 82.05% acc / 0.773 F1 |

**RoBERTa is the best-performing model across all three benchmarks**, confirming its robustness across different semantic similarity tasks and datasets.

### 5.2 Siamese Baseline Consistency

| Dataset | LSTM Accuracy/Pearson | GRU Accuracy/Pearson | Better Model |
|---------|:--------------------:|:--------------------:|:------------:|
| MRPC | 72.06% | 69.61% | LSTM |
| STS-B | 0.177 | 0.604 | GRU (**3.4× better**) |
| QQP | 67.60% | 64.40% | LSTM |

The Siamese models show inconsistent relative performance — LSTM is better on classification tasks (MRPC, QQP) while GRU is dramatically better on the regression task (STS-B). This suggests:
- **GRU's simpler gating mechanism** may be better suited to learning continuous similarity signals
- **LSTM may have training instability** on regression tasks with MSE loss, especially with frozen embeddings

### 5.3 DistilBERT vs BERT Across Datasets

| Dataset | BERT | DistilBERT | Winner | Gap |
|---------|:----:|:----------:|:------:|:---:|
| MRPC | 84.56% | **86.76%** | DistilBERT | +2.20 |
| STS-B | **0.884** | 0.863 | BERT | +0.022 |
| QQP | 81.20% | **81.95%** | DistilBERT | +0.75 |

DistilBERT wins 2 out of 3 benchmarks despite having 39% fewer parameters. BERT's advantage only manifests on STS-B's regression task, where the additional capacity may help model fine-grained similarity scores.

---

## 6. Static vs Contextualised Representations — Validated Hypothesis

The project proposal hypothesised that *"contextualised word representations enhance semantic comprehension compared to static embeddings"* and that *"fine-tuning pretrained language models improves performance on downstream NLP tasks."*

### Evidence Summary

| Comparison | Metric | Static (Best Siamese) | Contextualised (Best Transformer) | Gap |
|------------|--------|:---------------------:|:---------------------------------:|:---:|
| MRPC Accuracy | Acc. | 72.06% | 89.22% | **+17.16 pp** |
| MRPC F1 | F1 | 0.817 | 0.921 | **+0.104** |
| STS-B Correlation | Pearson | 0.604 | 0.886 | **+0.282** |
| QQP Accuracy | Acc. | 67.60% | 82.05% | **+14.45 pp** |
| QQP F1 | F1 | 0.636 | 0.773 | **+0.137** |

The gap is **consistent and substantial** across all tasks, datasets, and metrics — ranging from 14 to 28 percentage points. This conclusively validates the hypothesis.

### Why Static Embeddings Fail

1. **No context sensitivity**: GloVe assigns "bank" the same vector whether it means a financial institution or a river bank. Transformers produce context-dependent representations.
2. **No cross-sentence attention**: Siamese models encode sentences independently, then compare representations. Transformers with `[SEP]` tokens allow direct cross-attention between sentence pairs.
3. **Limited pre-training**: GloVe is trained on co-occurrence statistics only. BERT/RoBERTa are pre-trained on masked language modeling and (for BERT) next sentence prediction, learning deeper syntactic and semantic patterns.

---

## 7. Computational Cost vs Performance Trade-off

| Model | Params (M) | MRPC Acc | STS-B Pearson | QQP Acc | MRPC Inference (s) |
|-------|:----------:|:--------:|:-------------:|:-------:|:-------------------:|
| Siamese-LSTM | 1.5 | 72.06% | 0.177 | 67.60% | 0.11 |
| Siamese-GRU | 1.5 | 69.61% | 0.604 | 64.40% | 0.23 |
| DistilBERT | 67.0 | 86.76% | 0.863 | 81.95% | 1.43 |
| BERT | 109.5 | 84.56% | 0.884 | 81.20% | 2.46 |
| RoBERTa | 124.6 | 89.22% | 0.886 | 82.05% | 2.72 |

**Practical recommendations:**
- **Highest quality needed → RoBERTa**: Best across all benchmarks, but the most expensive at 125M params.
- **Balanced speed/quality → DistilBERT**: 86.76% MRPC accuracy at 1.9× the speed of BERT, with 39% fewer parameters. Best value proposition.
- **Latency-critical applications → Siamese-GRU**: 24× faster than transformers. Suitable only if ~70% accuracy is acceptable.
- **Similarity scoring without training data → SBERT**: Achieves 0.870 Pearson on STS-B with zero fine-tuning. Ideal for cold-start scenarios.

---

## 8. Limitations

1. **Limited fine-tuning epochs (2)**: Transformer models were only fine-tuned for 2 epochs. More epochs with learning rate scheduling and early stopping could improve results, particularly for BERT which may benefit from longer training.
2. **QQP subset**: Only 10K out of 364K training examples were used. Full-scale training would likely improve all models, especially transformers which are data-hungry.
3. **No hyperparameter tuning**: Fixed learning rates and batch sizes were used. Systematic tuning could narrow the BERT-DistilBERT gap.
4. **Small validation sets**: MRPC has only 408 validation samples, introducing metric variance.
5. **No cross-validation**: Results are from a single train/val split. Multiple runs would provide confidence intervals.

---

## 9. Generated Artifacts

### Results Files
| File | Dataset | Contents |
|------|---------|----------|
| `results/mrpc_results.json` | MRPC | Accuracy, F1, time, parameter counts |
| `results/sts_results.json` | STS-B | Pearson correlation, Spearman correlation |
| `results/qqp_results.json` | QQP | Accuracy, F1, time, parameter counts |

### Plots
| Plot | Path |
|------|------|
| MRPC Accuracy | `plots/mrpc/accuracy.png` |
| MRPC F1 Score | `plots/mrpc/f1.png` |
| MRPC Inference Time | `plots/mrpc/time.png` |
| MRPC Complexity Trade-off | `plots/mrpc/complexity_tradeoff.png` |
| STS-B Correlations | `plots/sts/sts_correlations.png` |
| QQP Accuracy | `plots/qqp/accuracy.png` |
| QQP F1 Score | `plots/qqp/f1.png` |
| QQP Inference Time | `plots/qqp/time.png` |
| QQP Complexity Trade-off | `plots/qqp/complexity_tradeoff.png` |
