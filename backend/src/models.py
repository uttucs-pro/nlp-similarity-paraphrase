from transformers import (
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    DistilBertForSequenceClassification
)

def get_bert():
    return BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )

def get_roberta():
    return RobertaForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=2
    )

def get_distilbert():
    return DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )


# --- Regression variants for STS-B (num_labels=1 → regression output) ---

def get_bert_regression():
    return BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=1
    )

def get_roberta_regression():
    return RobertaForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=1
    )

def get_distilbert_regression():
    return DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=1
    )
