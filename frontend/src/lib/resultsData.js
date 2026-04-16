export const datasetResults = {
  mrpc: {
    "Siamese-LSTM": {
      "accuracy": 0.6985294117647058,
      "f1": 0.7925801011804384,
      "time": 0.10100722312927246,
      "total_params": 1549926,
      "trainable_params": 309826
    },
    "Siamese-GRU": {
      "accuracy": 0.7328431372549019,
      "f1": 0.8192371475953566,
      "time": 0.23022103309631348,
      "total_params": 1491046,
      "trainable_params": 250946
    },
    "BERT": {
      "accuracy": 0.8848039215686274,
      "f1": 0.918825561312608,
      "time": 2.829314708709717,
      "total_params": 109483778,
      "trainable_params": 109483778
    },
    "RoBERTa": {
      "accuracy": 0.8823529411764706,
      "f1": 0.9157894736842105,
      "time": 2.7129971981048584,
      "total_params": 124647170,
      "trainable_params": 124647170
    },
    "DistilBERT": {
      "accuracy": 0.8357843137254902,
      "f1": 0.8834782608695653,
      "time": 1.4207789897918701,
      "total_params": 66955010,
      "trainable_params": 66955010
    }
  },
  qqp: {
    "Siamese-LSTM": {
      "accuracy": 0.676,
      "f1": 0.6241299303944315,
      "time": 0.5218544006347656,
      "total_params": 1817726,
      "trainable_params": 309826
    },
    "Siamese-GRU": {
      "accuracy": 0.644,
      "f1": 0.6363636363636364,
      "time": 1.139807939529419,
      "total_params": 1758846,
      "trainable_params": 250946
    },
    "BERT": {
      "accuracy": 0.812,
      "f1": 0.7655860349127181,
      "time": 13.36745810508728,
      "total_params": 109483778,
      "trainable_params": 109483778
    },
    "RoBERTa": {
      "accuracy": 0.8205,
      "f1": 0.772640911969601,
      "time": 12.934645891189575,
      "total_params": 124647170,
      "trainable_params": 124647170
    },
    "DistilBERT": {
      "accuracy": 0.8195,
      "f1": 0.7518900343642612,
      "time": 6.779628038406372,
      "total_params": 66955010,
      "trainable_params": 66955010
    }
  },
  sts: {
    "Siamese-LSTM": {
      "pearson": 0.6212981343269348,
      "spearman": 0.6502837461871585
    },
    "Siamese-GRU": {
      "pearson": 0.6761695146560669,
      "spearman": 0.6965090827109582
    },
    "BERT": {
      "pearson": 0.8909788727760315,
      "spearman": 0.8869297874479141
    },
    "RoBERTa": {
      "pearson": 0.9034045338630676,
      "spearman": 0.9002956471937638
    },
    "DistilBERT": {
      "pearson": 0.865139365196228,
      "spearman": 0.8620843226915993
    },
    "SBERT": {
      "pearson": 0.8696194526507806,
      "spearman": 0.8671631197908374
    }
  }
};
