# Load model directly
from transformers import pipeline
import pandas as pd
import numpy as np

def main():
    ## Load model

    classifier = pipeline("zero-shot-classification",model="MoritzLaurer/deberta-v3-base-zeroshot-v1")
    candidate_labels = ['Politics', 'Health', 'Finance', 'Travel', 'Food', 'Education', 'Environment', 'Fashion', 'Science', 'Sport', 'Technology', 'Entertainment']

    ## Prepare dataset

    test = pd.read_csv('data/test_shuffle.txt', sep = '.', header = None)
    txt = [t for t in test[0]]

    ## Inference

    out = [classifier(t, candidate_labels) for t in txt]

    # Get the first two labels with highest confidence and the confidence scores

    target1, confidence1, target2, confidence2 = [], [], [], []
    for dico in out:
        target1.append(dico['labels'][0])
        confidence1.append(dico['scores'][0])
        target2.append(dico['labels'][1])
        confidence2.append(dico['scores'][1])
        
    label_05 = []
    for i in range(len(target1)):
        if confidence1[i] - confidence2[i] < 0.05:
            label_05.append(target2[i])
        else:
            label_05.append(target1[i])

    label_1 = []
    for i in range(len(target1)):
        if confidence1[i] - confidence2[i] < 0.1:
            label_1.append(target2[i])
        else:
            label_1.append(target1[i])

    label_05_1 = []
    for i in range(len(target1)):
        if confidence1[i] - confidence2[i] >= 0.05 and confidence1[i] - confidence2[i] < 0.1:
            label_05_1.append(target2[i])
        else:
            label_05_1.append(target1[i])

    ## Save results

    df_target1st = pd.DataFrame({
        'Label' : target1,
        'ID' : np.arange(len(target1)),
    })

    df_target2nd = pd.DataFrame({
        'Label' : target2,
        'ID' : np.arange(len(target2)),
    })

    df_target_05 = pd.DataFrame({
        'Label' : label_05,
        'ID' : np.arange(len(label_05)),
    })

    df_target_1 = pd.DataFrame({
        'Label' : label_1,
        'ID' : np.arange(len(label_1)),
    })

    df_target_05_1 = pd.DataFrame({
        'Label' : label_05_1,
        'ID' : np.arange(len(label_05_1)),
    })

    df_target1st.to_csv('data/target1st.csv', index = False, header=1)
    # df_target2nd.to_csv('data/target2nd.csv', index = False, header=1)
    # df_target_05.to_csv('data/target2nd_05.csv', index = False, header=1)
    # df_target_1.to_csv('data/target2nd_1.csv', index = False, header=1)
    # df_target_05_1.to_csv('data/target2nd_05_1.csv', index = False, header=1)

if __name__ == "__main__":
    main()