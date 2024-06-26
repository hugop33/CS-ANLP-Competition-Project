# **Hand Labellisers** : Sentence-classification-ANLP
Axel VOYER, Hugo PLOTTU, Alexandre PETIT, Aymeric PALARIC

## Task
The task is to classify sentences into one of the following categories:
1. **Politics**
2. **Health**
3. **Finance**
4. **Travel**
5. **Food**
6. **Education**
7. **Environment**
8. **Fashion**
9. **Science**
10. **Sports**
11. **Technology**
12. **Entertainment**

## Dataset
The train dataset consists of 12 categories of sentences. Each category has 3 sentences. The test dataset has 1140 unlabelled sentences.

## Approach
We tried several approaches to classify the sentences into one of the 12 categories, as detailed in [this report](https://www.overleaf.com/read/jknfqbbrgymj#f7ad1f).

## Run
To run our best solution, which is zero-shot classification using deberta-large, run `python src/train/deberta_large.py`. Be sure to have a folder named `data` at the root of this project contaning the test set. The script will create a file `target1st.csv` in the `data` folder with the predictions of the model.