import torch
from torch.utils.data import Dataset
import json
from transformers import BertTokenizerFast

class TrainDataset(Dataset):
    def __init__(self, file):
        self.file = file
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.data = []
        self.labels = {}
        self.target = []
        with open(self.file, 'r') as f:
            data = json.load(f)
            for i, d in enumerate(data):
                self.labels[i] = d
                for sentence in data[d]:
                    self.data.append(sentence)
                    self.target.append(i)
        self.data = self.tokenizer(self.data, padding=True, truncation=True, return_tensors="pt")


    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.data['input_ids'][idx], self.data['attention_mask'][idx], self.data['token_type_ids'][idx], torch.tensor(self.target[idx])


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
        self.data = self.tokenizer(self.sentences, padding=True, truncation=True, return_tensors="pt")


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data['input_ids'][idx], self.data['attention_mask'][idx], self.data['token_type_ids'][idx]

if __name__=="__main__":
    dataset = TrainDataset('./data/train.json')
    print(len(dataset))
    print(dataset[3])
    print(dataset.labels)
    dataset = TestDataset('./data/test_shuffle.txt')
    print(len(dataset))
    print(dataset[3])