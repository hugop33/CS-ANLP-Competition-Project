# Load model directly
from transformers import pipeline
import pandas as pd
import numpy as np

## Load model

classifier = pipeline("zero-shot-classification",model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c")
candidate_labels = ['politics', 'health', 'finance', 'travel', 'travel', 'food', 'education', 'environment', 'fashion', 'science', 'sport', 'technology', 'entertainment']

## Example

text = "The role of credit scores in lending decisions is significant."
classifier(text, candidate_labels)

## Prepare dataset

test = pd.read_csv('./data/test_shuffle.txt', sep = '.', header = None)
txt = [t for t in test[0]]

## Inference

out = [classifier(t, candidate_labels) for t in txt[300:]]
target = [dico['labels'][np.argmax(dico['scores'])] for dico in out]

## Save results

df_target = pd.DataFrame({
    'Label' : target
})
df_target['ID'] = df_target.index
df_target=df_target.reindex(columns=["ID", "Label"])
df_target.to_csv('target_submit.csv', index = False, header=1)