import torch
from torch.utils.data import DataLoader
from src.dataloader.dataloading import TrainDataset, TestDataset
from src.models.classifier import Classifier
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import random_split
from tqdm import tqdm, trange
from sklearn.feature_extraction.text import TfidfVectorizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model, dataloader, criterion, *optimizers, print=True):
    model.train()
    total_loss = 0
    with tqdm(total=len(dataloader), disable=not print) as pbar:
        for (embedding, target) in dataloader:
            embedding = embedding.to(DEVICE)
            target = target.to(DEVICE)
            for optimizer in optimizers:
                optimizer.zero_grad()
            output = model(embedding)
            loss = criterion(output, target)
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            pbar.update(1)

    return total_loss/len(dataloader)


def eval_epoch(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for (embedding, target) in dataloader:
            embedding = embedding.to(DEVICE)
            target = target.to(DEVICE)
            output = model(embedding)
            loss = criterion(output, target)
            total_loss += loss.item()
            total_correct += (output.argmax(1) == target).sum().item()
    return total_loss/len(dataloader), total_correct/len(dataloader.dataset)


def train(model, train_dataloader, val_dataloader, criterion, num_epochs, *optimizers, print_cp=1):
    for epoch in range(num_epochs):
        if (epoch+1) % print_cp == 0:
            print(f'Epoch {epoch+1}/{num_epochs}')
        train_loss = train_epoch(model, train_dataloader, criterion, *optimizers, print=False if (epoch+1) % print_cp != 0 else True)
        # val_loss, val_acc = eval_epoch(model, val_dataloader, criterion)
        if (epoch+1) % print_cp == 0:
            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'Train loss: {train_loss:.4f}')
            # print(f'Val loss: {val_loss:.4f}')
            # print(f'Val accuracy: {val_acc:.4f}')
            print('-------------------')
    
    # predict on all test data
    model.eval()
    predictions = []
    with torch.no_grad():
        for embedding in val_dataloader:
            embedding = embedding.to(DEVICE)
            output = model(embedding)
            predictions.append(output.argmax(1))
    torch.save(model.state_dict(), './model_weights/clf_dense.pth')
    return torch.cat(predictions).numpy()


if __name__=="__main__":
    import pandas as pd

    train_dataset = TrainDataset('./data/train.json')
    labels = train_dataset.labels
    vectorizer = TfidfVectorizer()
    train_embeddings = vectorizer.fit_transform(train_dataset.sentences)
    train_embeddings = torch.tensor(train_embeddings.toarray(), dtype=torch.float32)
    targets = train_dataset.target
    train_dataloader = DataLoader(list(zip(train_embeddings, targets)), batch_size=4, shuffle=True)
    val_dataset = TestDataset('./data/test_shuffle.txt')
    val_embeddings = vectorizer.transform(val_dataset.sentences)
    val_embeddings = torch.tensor(val_embeddings.toarray(), dtype=torch.float32)
    val_dataloader = DataLoader(val_embeddings, batch_size=32, shuffle=False)
    model = Classifier(train_embeddings.shape[1], 256, 12).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    test_preds = train(model, train_dataloader, val_dataloader, criterion, 1000, optimizer, print_cp=100)
    test_df = pd.DataFrame()
    test_df["pred"] = test_preds
    test_df['Label'] = test_df['pred'].apply(lambda x: labels[x])
    test_df["ID"] = test_df.index
    test_df.drop(columns=["pred"], inplace=True)
    test_df.to_csv('./data/test_tfidf.csv', index=False)
