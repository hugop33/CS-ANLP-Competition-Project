import torch
from torch.utils.data import DataLoader
from src.dataloader.dataloading import TrainDataset, TestDataset, AdditionalDataset
from src.models.bert import BertEmbedding, BertClassifier
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import random_split
from tqdm import tqdm
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model, dataloader, criterion, *optimizers):
    model.train()
    total_loss = 0
    with tqdm(total=len(dataloader)) as pbar:
        for input_ids, attention_mask, token_type_ids, target in dataloader:
            for optimizer in optimizers:
                optimizer.zero_grad()
            input_ids, attention_mask, token_type_ids, target = input_ids.to(DEVICE), attention_mask.to(DEVICE), token_type_ids.to(DEVICE), target.to(DEVICE)
            output = model(input_ids, attention_mask, token_type_ids)
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
        for input_ids, attention_mask, token_type_ids, target in dataloader:
            input_ids, attention_mask, token_type_ids, target = input_ids.to(DEVICE), attention_mask.to(DEVICE), token_type_ids.to(DEVICE), target.to(DEVICE)
            output = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(output, target)
            total_loss += loss.item()
            total_correct += (output.argmax(1) == target).sum().item()
    return total_loss/len(dataloader), total_correct/len(dataloader.dataset)


def train(model, train_dataloader, val_dataloader, criterion, num_epochs, *optimizers):
    train_dataset = train_dataloader.dataset
    train_dataset, test_dataset = random_split(train_dataset, [int(0.8*len(train_dataset)), len(train_dataset)-int(0.8*len(train_dataset))])
    train_dataloader = DataLoader(train_dataset, batch_size=train_dataloader.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=val_dataloader.batch_size, shuffle=False)
    model.to(DEVICE)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        train_loss = train_epoch(model, train_dataloader, criterion, *optimizers)
        val_loss, val_acc = eval_epoch(model, test_dataloader, criterion)
        print(f'Train loss: {train_loss:.4f}')
        print(f'Val loss: {val_loss:.4f}')
        print(f'Val accuracy: {val_acc:.4f}')
        print('-------------------')
    
    torch.save(model.state_dict(), './model_weights/full_semi_bert_clf_augmented.pth')
    return predictions(model, val_dataloader, train_dataset)

def predictions(model, test_dataset):
    model.eval()
    model.to(DEVICE)
    predictions = []
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids in test_dataset:
            input_ids, attention_mask, token_type_ids = input_ids.to(DEVICE), attention_mask.to(DEVICE), token_type_ids.to(DEVICE)
            output = model(input_ids, attention_mask, token_type_ids)
            predictions.extend(output.argmax(1).cpu().tolist())
    return predictions

if __name__=="__main__":
    import pandas as pd

    train_dataset = TrainDataset('./data/augmented_semi.json')
    # additional_dataset = AdditionalDataset('./data/labelled_newscatcher_dataset.csv')
    # train_dataset.extend(additional_dataset)
    labels = train_dataset.labels
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataset = TestDataset('./data/test_shuffle.txt')
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    model = BertClassifier(freeze_bert=False)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = Adam(model.parameters(), lr=1e-3)
    # lin_optimizer = Adam(model.fc.parameters(), lr=5e-3)
    # bert_optimizer = Adam(model.bert.parameters(), lr=1e-5)
    # test_preds = train(model, train_dataloader, val_dataloader, criterion, 5, lin_optimizer, bert_optimizer)
    # test_df = pd.DataFrame()
    # test_df["pred"] = test_preds
    # test_df['Label'] = test_df['pred'].apply(lambda x: labels[x])
    # test_df["ID"] = test_df.index
    # test_df.drop(columns=["pred"], inplace=True)
    # test_df.to_csv('./data/full_semi_bert_augmented_data.csv', index=False)
    model.load_state_dict(torch.load('./model_weights/semi_bert_clf_augmented.pth'))
    test_preds = predictions(model, val_dataloader)
    test_df = pd.DataFrame()
    test_df["pred"] = test_preds
    test_df['Label'] = test_df['pred'].apply(lambda x: labels[x])
    test_df["ID"] = test_df.index
    test_df.drop(columns=["pred"], inplace=True)
    partial_df = pd.read_csv('./data/partial_naive.csv')
    partial_df[partial_df['Label'] == 'Others'] = test_df
    partial_df.to_csv('./data/semi_naive_augmented_bert.csv', index=False)
