# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("sileod/deberta-v3-base-tasksource-nli")
model = AutoModelForSequenceClassification.from_pretrained("sileod/deberta-v3-base-tasksource-nli")


classifier = pipeline("zero-shot-classification",model="sileod/deberta-v3-base-tasksource-nli")

text = "The role of credit scores in lending decisions is significant."
candidate_labels = ['politics', 'health', 'finance', 'travel', 'travel', 'food', 'education', 'environment', 'fashion', 'science', 'sport', 'technology', 'entertainment']
classifier(text, candidate_labels)


test = pd.read_csv('./data/test_shuffle.txt', sep = '.', header = None)
txt = [t for t in test[0]]
out = [classifier(t, candidate_labels) for t in txt[300:]]
target = [dico['labels'][np.argmax(dico['scores'])] for dico in out]