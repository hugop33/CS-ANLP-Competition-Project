from torch import nn
import torch
from transformers import BertModel

class BertEmbedding(nn.Module):
    def __init__(self):
        super(BertEmbedding, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.bert(input_ids, attention_mask, token_type_ids)
    

class BertClassifier(nn.Module):
    def __init__(self, num_classes=12, freeze_bert=True):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # freeze bert
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.fc = nn.Linear(768, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids, attention_mask, token_type_ids)
        mean_embedding = torch.mean(output[0], dim=1)
        return self.softmax(self.fc(mean_embedding))
    

class SiameseBert(nn.Module):
    def __init__(self, freeze_bert=True):
        """
        Binary classification model using Siamese Bert : determines if two sentences are of the same class or not
        - Takes two input sentences
        - Passes them through a shared bert model
        - Computes the difference between the pooled outputs
        - Passes the difference through a linear layer
        """
        super(SiameseBert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # freeze bert
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.fc = nn.Sequential(nn.Linear(3*768, 512),
                                nn.ReLU(),
                                nn.Linear(512, 1))
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, sentence1, sentence2):
        input_ids1, attention_mask1, token_type_ids1 = sentence1
        input_ids2, attention_mask2, token_type_ids2 = sentence2
        output1 = self.bert(input_ids1, attention_mask1, token_type_ids1)
        output2 = self.bert(input_ids2, attention_mask2, token_type_ids2)
        pooled1 = torch.mean(output1[0], dim=1)
        pooled2 = torch.mean(output2[0], dim=1)
        diff = torch.abs(pooled1 - pooled2)
        aggregate = torch.cat([pooled1, pooled2, diff], dim=1)
        return self.sigmoid(self.fc(aggregate))


if __name__=="__main__":
    from torch.utils.data import DataLoader
    from src.dataloader.dataloading import TrainDataset, TestDataset, LabelDataset
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    from transformers import pipeline


    ###### Visualize embeddings
    # dataset = TrainDataset('./data/train.json')
    # labels = dataset.labels
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    # model = BertEmbedding()
    # model.eval()
    # embeddings = []
    # targets = []
    # with torch.no_grad():
    #     for input_ids, attention_mask, token_type_ids, target in dataloader:
    #         targets.append(target)
    #         output = model(input_ids, attention_mask, token_type_ids)
    #         embeddings.append(output[0])
    # # print(len(embeddings))

    # # tsne = TSNE(n_components=2)
    # # embeddings = torch.cat(embeddings, dim=0)
    # # embeddings = embeddings.squeeze(1)
    # # embeddings = embeddings[:,0,:]
    # # print(embeddings.size())
    # # embeddings_t = tsne.fit_transform(embeddings)
    # # print(embeddings_t.shape)
    # # plt.scatter(embeddings_t[:,0], embeddings_t[:,1], c=torch.cat(targets).numpy())
    # # plt.show()

    # # pca = PCA()
    # # embeddings_p = pca.fit_transform(embeddings)[:,:2]
    # # print(pca.explained_variance_ratio_)
    # # plt.scatter(embeddings_p[:,0], embeddings_p[:,1], c=torch.cat(targets).numpy())
    # # plt.show()
    # train_embeddings = torch.cat(embeddings, dim=0)
    # train_embeddings = train_embeddings.squeeze(1)
    # train_embeddings = train_embeddings[:,0,:]

    # test_dataset = TestDataset('./data/test_shuffle.txt')
    # test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    # test_embeddings = []
    # with torch.no_grad():
    #     for input_ids, attention_mask, token_type_ids in test_dataloader:
    #         output = model(input_ids, attention_mask, token_type_ids)
    #         test_embeddings.append(output[0])
    # test_embeddings = torch.cat(test_embeddings, dim=0)
    # test_embeddings = test_embeddings.squeeze(1)
    # test_embeddings = test_embeddings[:,0,:]

    # from sklearn.neighbors import KNeighborsClassifier
    # import pandas as pd
    # # pick a random sample in test set
    # from numpy import random

    # classifier = KNeighborsClassifier(n_neighbors=3)
    # classifier.fit(train_embeddings, torch.cat(targets).numpy())
    # baseline_df = pd.DataFrame()
    # baseline_df['sentence'] = test_dataset.sentences
    # baseline_df['pred'] = classifier.predict(test_embeddings)
    # baseline_df["labels"] = baseline_df["pred"].apply(lambda x: labels[x])
    # baseline_df.to_csv('./data/baseline_preds.csv', index=False)


    # label_dataset = LabelDataset('./data/train.json')
    # label_dataloader = DataLoader(label_dataset, batch_size=8, shuffle=False)
    # label_embeddings = []
    # with torch.no_grad():
    #     for input_ids, attention_mask, token_type_ids, _ in label_dataloader:
    #         output = model(input_ids, attention_mask, token_type_ids)
    #         label_embeddings.append(output[0])
    # label_embeddings = torch.cat(label_embeddings, dim=0)
    # label_embeddings = label_embeddings.squeeze(1)
    # label_embeddings = label_embeddings[:,0,:]

    # proximity_df = pd.DataFrame()
    # proximity_df['sentence'] = test_dataset.sentences
    # #prediction is the closest label in the embedding space
    # proximity_df['pred'] = torch.argmin(torch.cdist(test_embeddings, label_embeddings, p=2, compute_mode='use_mm_for_euclid_dist'), dim=1)
    # proximity_df["labels"] = proximity_df["pred"].apply(lambda x: labels[x])
    # proximity_df.to_csv('./data/proximity_preds.csv', index=False)
    # ########################################

    dataset = TrainDataset('./data/train.json')
    # print(dataset.labels)
    classifier = pipeline("zero-shot-classification",model="sileod/deberta-v3-base-tasksource-nli")
    labels = list(dataset.labels.values())
    print(labels)
    lbl_mapping = {v: k for k, v in dataset.labels.items()}

    preds = classifier(dataset.sentences, labels, multi_label=False)
    print(preds)
    accuracy = 0
    for pred, target in zip(preds, dataset.target):
        lbl = pred['labels'][0]
        k = lbl_mapping[lbl]
        if k == target:
            accuracy += 1
        else:
            print(f"Error with sentence: {pred['sequence']}")
            print(f"Predicted: {lbl}, Target: {dataset.labels[target]}")
    print(accuracy/len(dataset.target))
