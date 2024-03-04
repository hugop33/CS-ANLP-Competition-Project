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
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids, attention_mask, token_type_ids)
        cls_embedding = output[0][:,0,:]
        return self.fc(cls_embedding)


if __name__=="__main__":
    from torch.utils.data import DataLoader
    from src.dataloader.dataloading import TrainDataset, TestDataset, LabelDataset
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    dataset = TrainDataset('./data/train.json')
    labels = dataset.labels
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = BertEmbedding()
    model.eval()
    embeddings = []
    targets = []
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids, target in dataloader:
            targets.append(target)
            output = model(input_ids, attention_mask, token_type_ids)
            embeddings.append(output[0])
    # print(len(embeddings))

    # tsne = TSNE(n_components=2)
    # embeddings = torch.cat(embeddings, dim=0)
    # embeddings = embeddings.squeeze(1)
    # embeddings = embeddings[:,0,:]
    # print(embeddings.size())
    # embeddings_t = tsne.fit_transform(embeddings)
    # print(embeddings_t.shape)
    # plt.scatter(embeddings_t[:,0], embeddings_t[:,1], c=torch.cat(targets).numpy())
    # plt.show()

    # pca = PCA()
    # embeddings_p = pca.fit_transform(embeddings)[:,:2]
    # print(pca.explained_variance_ratio_)
    # plt.scatter(embeddings_p[:,0], embeddings_p[:,1], c=torch.cat(targets).numpy())
    # plt.show()
    train_embeddings = torch.cat(embeddings, dim=0)
    train_embeddings = train_embeddings.squeeze(1)
    train_embeddings = train_embeddings[:,0,:]

    test_dataset = TestDataset('./data/test_shuffle.txt')
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    test_embeddings = []
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids in test_dataloader:
            output = model(input_ids, attention_mask, token_type_ids)
            test_embeddings.append(output[0])
    test_embeddings = torch.cat(test_embeddings, dim=0)
    test_embeddings = test_embeddings.squeeze(1)
    test_embeddings = test_embeddings[:,0,:]

    from sklearn.neighbors import KNeighborsClassifier
    import pandas as pd
    # pick a random sample in test set
    from numpy import random

    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(train_embeddings, torch.cat(targets).numpy())
    # rand_idx = random.randint(0, len(test_dataset))
    # with torch.no_grad():
    #     sample_input_ids, sample_attention_mask, sample_token_type_ids = test_dataset[rand_idx]
    #     sample_input_ids = sample_input_ids.unsqueeze(0)
    #     sample_attention_mask = sample_attention_mask.unsqueeze(0)
    #     sample_token_type_ids = sample_token_type_ids.unsqueeze(0)
    #     sample_embedding = model(sample_input_ids, sample_attention_mask, sample_token_type_ids)[0]
    #     sample_embedding = sample_embedding.squeeze(1)
    #     sample_embedding = sample_embedding[:,0,:]
    # pred = classifier.predict(sample_embedding)
    # print(test_dataset.sentences[rand_idx])
    # print(dataset.labels[pred[0]])
    baseline_df = pd.DataFrame()
    baseline_df['sentence'] = test_dataset.sentences
    baseline_df['pred'] = classifier.predict(test_embeddings)
    baseline_df["labels"] = baseline_df["pred"].apply(lambda x: labels[x])
    baseline_df.to_csv('./data/baseline_preds.csv', index=False)


    label_dataset = LabelDataset('./data/train.json')
    label_dataloader = DataLoader(label_dataset, batch_size=8, shuffle=False)
    label_embeddings = []
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids, _ in label_dataloader:
            output = model(input_ids, attention_mask, token_type_ids)
            label_embeddings.append(output[0])
    label_embeddings = torch.cat(label_embeddings, dim=0)
    label_embeddings = label_embeddings.squeeze(1)
    label_embeddings = label_embeddings[:,0,:]

    proximity_df = pd.DataFrame()
    proximity_df['sentence'] = test_dataset.sentences
    #prediction is the closest label in the embedding space
    proximity_df['pred'] = torch.argmin(torch.cdist(test_embeddings, label_embeddings, p=2, compute_mode='use_mm_for_euclid_dist'), dim=1)
    proximity_df["labels"] = proximity_df["pred"].apply(lambda x: labels[x])
    proximity_df.to_csv('./data/proximity_preds.csv', index=False)


