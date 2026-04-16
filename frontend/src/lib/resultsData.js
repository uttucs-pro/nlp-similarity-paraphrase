/**
 * Static benchmark results organized by dataset → variant → model.
 *
 * "base" = pre-trained model, no fine-tuning (embedding cosine similarity).
 * "tuned" = fine-tuned on the specific dataset.
 *
 * Base results will be populated after running base evaluation.
 * Tuned results come from actual training runs.
 */
export const datasetResults = {
  mrpc: {
    task: "paraphrase_detection",
    base: {
      "Siamese-LSTM": { accuracy: 0, f1: 0, time: 0, total_params: 0, trainable_params: 0 },
      "Siamese-GRU":  { accuracy: 0, f1: 0, time: 0, total_params: 0, trainable_params: 0 },
      "BERT":         { accuracy: 0, f1: 0, time: 0, total_params: 109482240, trainable_params: 0 },
      "RoBERTa":      { accuracy: 0, f1: 0, time: 0, total_params: 124645632, trainable_params: 0 },
      "DistilBERT":   { accuracy: 0, f1: 0, time: 0, total_params: 66362880, trainable_params: 0 },
    },
    tuned: {
      "Siamese-LSTM": { accuracy: 0.6985, f1: 0.7926, time: 0.101, total_params: 1549926, trainable_params: 309826 },
      "Siamese-GRU":  { accuracy: 0.7328, f1: 0.8192, time: 0.230, total_params: 1491046, trainable_params: 250946 },
      "BERT":         { accuracy: 0.8848, f1: 0.9188, time: 2.829, total_params: 109483778, trainable_params: 109483778 },
      "RoBERTa":      { accuracy: 0.8824, f1: 0.9158, time: 2.713, total_params: 124647170, trainable_params: 124647170 },
      "DistilBERT":   { accuracy: 0.8358, f1: 0.8835, time: 1.421, total_params: 66955010, trainable_params: 66955010 },
    },
  },
  qqp: {
    task: "paraphrase_detection",
    base: {
      "Siamese-LSTM": { accuracy: 0, f1: 0, time: 0, total_params: 0, trainable_params: 0 },
      "Siamese-GRU":  { accuracy: 0, f1: 0, time: 0, total_params: 0, trainable_params: 0 },
      "BERT":         { accuracy: 0, f1: 0, time: 0, total_params: 109482240, trainable_params: 0 },
      "RoBERTa":      { accuracy: 0, f1: 0, time: 0, total_params: 124645632, trainable_params: 0 },
      "DistilBERT":   { accuracy: 0, f1: 0, time: 0, total_params: 66362880, trainable_params: 0 },
    },
    tuned: {
      "Siamese-LSTM": { accuracy: 0.676, f1: 0.6241, time: 0.522, total_params: 1817726, trainable_params: 309826 },
      "Siamese-GRU":  { accuracy: 0.644, f1: 0.6364, time: 1.140, total_params: 1758846, trainable_params: 250946 },
      "BERT":         { accuracy: 0.812, f1: 0.7656, time: 13.367, total_params: 109483778, trainable_params: 109483778 },
      "RoBERTa":      { accuracy: 0.8205, f1: 0.7726, time: 12.935, total_params: 124647170, trainable_params: 124647170 },
      "DistilBERT":   { accuracy: 0.8195, f1: 0.7519, time: 6.780, total_params: 66955010, trainable_params: 66955010 },
    },
  },
  stsb: {
    task: "semantic_similarity",
    base: {
      "Siamese-LSTM": { pearson: 0, spearman: 0 },
      "Siamese-GRU":  { pearson: 0, spearman: 0 },
      "BERT":         { pearson: 0, spearman: 0 },
      "RoBERTa":      { pearson: 0, spearman: 0 },
      "DistilBERT":   { pearson: 0, spearman: 0 },
    },
    tuned: {
      "Siamese-LSTM": { pearson: 0.6213, spearman: 0.6503 },
      "Siamese-GRU":  { pearson: 0.6762, spearman: 0.6965 },
      "BERT":         { pearson: 0.8910, spearman: 0.8869 },
      "RoBERTa":      { pearson: 0.9034, spearman: 0.9003 },
      "DistilBERT":   { pearson: 0.8651, spearman: 0.8621 },
      "SBERT":        { pearson: 0.8696, spearman: 0.8672 },
    },
  },
};
