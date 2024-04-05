import torch
from torch.utils.data import Dataset
import json
from transformers import BertTokenizerFast
import pandas as pd

class TrainDataset(Dataset):
    def __init__(self, file, pad_len=128):
        self.file = file
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.sentences = []
        self.labels = {}
        self.target = []
        self.pad_len = pad_len
        with open(self.file, 'r') as f:
            data = json.load(f)
            for i, d in enumerate(data):
                self.labels[i] = d
                for sentence in data[d]:
                    self.sentences.append(sentence)
                    self.target.append(i)
        self.data = self.tokenizer(self.sentences, padding="max_length", max_length=pad_len, return_tensors="pt")
        self.pad_len = self.data['input_ids'].size(1)


    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.data['input_ids'][idx], self.data['attention_mask'][idx], self.data['token_type_ids'][idx], torch.tensor(self.target[idx])

    def get_batchlbl(self, label_idx):
        """
        get all sentences with a specific label, given by its index
        """
        idxs = [i for i, l in enumerate(self.target) if l == label_idx]
        return self.data['input_ids'][idxs], self.data['attention_mask'][idxs], self.data['token_type_ids'][idxs], torch.tensor(self.target)[idxs]
    
    def extend(self, dataset:Dataset):
        self.data['input_ids'] = torch.cat((self.data['input_ids'], dataset.data['input_ids']))
        self.data['attention_mask'] = torch.cat((self.data['attention_mask'], dataset.data['attention_mask']))
        self.data['token_type_ids'] = torch.cat((self.data['token_type_ids'], dataset.data['token_type_ids']))
        self.target = self.target + dataset.target
        return self


class SiameseDataset(Dataset):
    def __init__(self, file, pad_len=128):
        self.file = file
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.sentences = []
        self.labels = {}
        self.target = []
        self.pad_len = pad_len
        with open(self.file, 'r') as f:
            data = json.load(f)
            for i, d in enumerate(data):
                self.labels[i] = d
                for sentence in data[d]:
                    self.sentences.append(sentence)
                    self.target.append(i)
        self.data = self.tokenizer(self.sentences, padding="max_length", max_length=pad_len, return_tensors="pt")
        self.pad_len = self.data['input_ids'].size(1)
        self.pairs = []
        for i in range(len(self.target)):
            for j in range(i+1, len(self.target)):
                self.pairs.append((i, j, 1 if self.target[i] == self.target[j] else 0))
        self.len = len(self.pairs)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        i, j, target = self.pairs[idx]
        return (self.data['input_ids'][i], self.data['attention_mask'][i], self.data['token_type_ids'][i]), (self.data['input_ids'][j], self.data['attention_mask'][j], self.data['token_type_ids'][j]), torch.tensor(target)


class TestDataset(Dataset):
    def __init__(self, file):
        self.file = file
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.sentences = []
        with open(self.file, 'r') as f:
            for l in f.readlines():
                if l.strip() != '':
                    self.sentences.append(l.strip())
        self.len = len(self.sentences)
        self.data = self.tokenizer(self.sentences, padding="longest", return_tensors="pt")


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data['input_ids'][idx], self.data['attention_mask'][idx], self.data['token_type_ids'][idx]


class LabelDataset(Dataset):
    def __init__(self, file):
        self.file = file
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.labels = []
        self.tokens = []
        with open(self.file, 'r') as f:
            data = json.load(f)
            for d in data:
                self.labels.append(d)
        self.tokens = self.tokenizer(self.labels, padding="longest", return_tensors="pt")
        self.len = len(self.labels)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.tokens['input_ids'][idx], self.tokens['attention_mask'][idx], self.tokens['token_type_ids'][idx], self.labels[idx]


class AdditionalDataset(Dataset):
    def __init__(self, file, pad_len=128):
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.topic2label = {
            "TECHNOLOGY": 10,
            "SCIENCE": 8,
            "BUSINESS": 3,
            "SPORTS": 9,
            "ENTERTAINMENT": 11,
            "HEALTH": 1
        }
        df = pd.read_csv(file, sep=';')
        self.pad_len = pad_len
        self.texts = []
        self.target = []
        for i, row in df.iterrows():
            if row['topic'] in self.topic2label:
                self.texts.append(row['title'])
                self.target.append(self.topic2label[row['topic']])
        self.data = self.tokenizer(self.texts, padding="max_length", max_length=self.pad_len, return_tensors="pt")
    
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        return self.data['input_ids'][idx], self.data['attention_mask'][idx], self.data['token_type_ids'][idx], self.target[idx]

if __name__=="__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    # train_dataset = TrainDataset('./data/train.json')
    # print(len(dataset))
    # print(dataset[3])
    # print(train_dataset.labels)
    # dataset = TestDataset('./data/test_shuffle.txt')
    # print(len(dataset))
    # print(dataset[3])

    # dataset = LabelDataset('./data/train.json')
    # print(len(dataset))
    # print(dataset[3])

    # dataset = AdditionalDataset('data/labelled_newscatcher_dataset.csv')
    # print(len(dataset))
    # print(dataset[3])

    # train_dataset.extend(dataset)
    # print(len(train_dataset))

    dataset = SiameseDataset('./data/augmented.json')
    print(len(dataset))
    print(dataset[2])
    print(dataset.target[0], dataset.target[3])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
