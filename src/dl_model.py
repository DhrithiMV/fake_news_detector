# src/dl_model.py
from transformers import pipeline

# Load once
_dl_classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def dl_predict(text: str):
    """
    Deep Learning prediction using DistilBERT
    Returns label and confidence
    """
    result = _dl_classifier(text[:512])[0]
    return result["label"], result["score"]
