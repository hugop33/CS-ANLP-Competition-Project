from torch import nn
import torch
from transformers import BertModel

class BertEmbedding(nn.Module):
    def __init__(self):
        super(BertEmbedding, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.bert(input_ids, attention_mask, token_type_ids)


if __name__=="__main__":
    from torch.utils.data import DataLoader
    from src.dataloader.dataloading import TrainDataset, TestDataset
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    dataset = TrainDataset('./data/train.json')
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
    # test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    # test_embeddings = []
    # with torch.no_grad():
    #     for input_ids, attention_mask, token_type_ids in test_dataloader:
    #         output = model(input_ids, attention_mask, token_type_ids)
    #         test_embeddings.append(output[0])
    # test_embeddings = torch.cat(test_embeddings, dim=0)
    # test_embeddings = test_embeddings.squeeze(1)
    # test_embeddings = test_embeddings[:,0,:]

    from sklearn.neighbors import KNeighborsClassifier
    # pick a random sample in test set
    from numpy import random

    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(train_embeddings, torch.cat(targets).numpy())
    rand_idx = random.randint(0, len(test_dataset))
    with torch.no_grad():
        sample_input_ids, sample_attention_mask, sample_token_type_ids = test_dataset[rand_idx]
        sample_input_ids = sample_input_ids.unsqueeze(0)
        sample_attention_mask = sample_attention_mask.unsqueeze(0)
        sample_token_type_ids = sample_token_type_ids.unsqueeze(0)
        sample_embedding = model(sample_input_ids, sample_attention_mask, sample_token_type_ids)[0]
        sample_embedding = sample_embedding.squeeze(1)
        sample_embedding = sample_embedding[:,0,:]
    pred = classifier.predict(sample_embedding)
    print(test_dataset.sentences[rand_idx])
    print(dataset.labels[pred[0]])



