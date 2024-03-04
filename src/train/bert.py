import torch
from torch.utils.data import DataLoader
from src.dataloader.dataloading import TrainDataset, TestDataset
from src.models.bert import BertEmbedding, BertClassifier
from torch.optim import Adam
import torch.nn as nn


def train_epoch(model, dataloader, criterion, *optimizers):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, token_type_ids, target in dataloader:
        for optimizer in optimizers:
            optimizer.zero_grad()
        output = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(output, target)
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        total_loss += loss.item()
    return total_loss/len(dataloader)


def eval_epoch(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids, target in dataloader:
            output = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(output, target)
            total_loss += loss.item()
            total_correct += (output.argmax(1) == target).sum().item()
    return total_loss/len(dataloader), total_correct/len(dataloader.dataset)


def train(model, train_dataloader, val_dataloader, criterion, num_epochs, *optimizers):
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_dataloader, criterion, *optimizers)
        val_loss, val_acc = eval_epoch(model, train_dataloader, criterion)
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train loss: {train_loss:.4f}')
        print(f'Val loss: {val_loss:.4f}')
        print(f'Val accuracy: {val_acc:.4f}')
        print('-------------------')
    
    # predict on all test data
    model.eval()
    predictions = []
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids in val_dataloader:
            output = model(input_ids, attention_mask, token_type_ids)
            predictions.append(output.argmax(1))
    torch.save(model.state_dict(), './model_weights/bert_clf.pth')
    return torch.cat(predictions).numpy()


if __name__=="__main__":
    import pandas as pd

    train_dataset = TrainDataset('./data/train.json')
    labels = train_dataset.labels
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataset = TestDataset('./data/test_shuffle.txt')
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    model = BertClassifier(freeze_bert=False)
    criterion = nn.CrossEntropyLoss()
    lin_optimizer = Adam(model.fc.parameters(), lr=1e-2)
    bert_optimizer = Adam(model.bert.parameters(), lr=1e-5)
    test_preds = train(model, train_dataloader, val_dataloader, criterion, 5, lin_optimizer, bert_optimizer)
    test_df = pd.DataFrame()
    test_df['sentence'] = val_dataset.sentences
    test_df['pred'] = test_preds
    test_df['label'] = test_df['pred'].apply(lambda x: labels[x])
    test_df.to_csv('./data/test_preds_bert.csv', index=False)
