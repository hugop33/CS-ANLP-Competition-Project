import torch
from torch.utils.data import DataLoader
from src.dataloader.dataloading import SiameseDataset, TestDataset, TrainDataset
from src.models.bert import SiameseBert
from torch.optim import Adam
import torch.nn as nn
import numpy as np
from torch.utils.data import random_split
from tqdm import tqdm
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model, dataloader, criterion, *optimizers):
    model.train()
    total_loss = 0
    with tqdm(total=len(dataloader)) as pbar:
        for sent1, sent2, target in dataloader:
            for optimizer in optimizers:
                optimizer.zero_grad()
            input_ids1, attention_mask1, token_type_ids1 = sent1
            input_ids2, attention_mask2, token_type_ids2 = sent2
            input_ids1, attention_mask1, token_type_ids1, input_ids2, attention_mask2, token_type_ids2, target = input_ids1.to(DEVICE), attention_mask1.to(DEVICE), token_type_ids1.to(DEVICE), input_ids2.to(DEVICE), attention_mask2.to(DEVICE), token_type_ids2.to(DEVICE), target.to(DEVICE)
            output = model((input_ids1, attention_mask1, token_type_ids1), (input_ids2, attention_mask2, token_type_ids2))
            output = output.squeeze(1).float()
            target = target.float()
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
        for sent1, sent2, target in dataloader:
            input_ids1, attention_mask1, token_type_ids1 = sent1
            input_ids2, attention_mask2, token_type_ids2 = sent2
            input_ids1, attention_mask1, token_type_ids1, input_ids2, attention_mask2, token_type_ids2, target = input_ids1.to(DEVICE), attention_mask1.to(DEVICE), token_type_ids1.to(DEVICE), input_ids2.to(DEVICE), attention_mask2.to(DEVICE), token_type_ids2.to(DEVICE), target.to(DEVICE)
            output = model((input_ids1, attention_mask1, token_type_ids1), (input_ids2, attention_mask2, token_type_ids2))
            output = output.squeeze(1).float()
            target = target.float()
            loss = criterion(output, target)
            total_loss += loss.item()
            # binary classification (1 output node)
            approx = output.round()
            total_correct += (approx == target).sum().item()
    return total_loss/len(dataloader), total_correct/len(dataloader.dataset)


def train(model, train_dataloader, val_dataset, ref_dataset, criterion, num_epochs, *optimizers):
    train_dataset = train_dataloader.dataset
    train_dataset, test_dataset = random_split(train_dataset, [int(0.8*len(train_dataset)), len(train_dataset)-int(0.8*len(train_dataset))])
    train_dataloader = DataLoader(train_dataset, batch_size=train_dataloader.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    model.to(DEVICE)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        train_loss = train_epoch(model, train_dataloader, criterion, *optimizers)
        val_loss, val_acc = eval_epoch(model, test_dataloader, criterion)
        print(f'Train loss: {train_loss:.4f}')
        print(f'Val loss: {val_loss:.4f}')
        print(f'Val accuracy: {val_acc:.4f}')
        print('-------------------')
    torch.save(model.state_dict(), './model_weights/sentencebert_augmented_semi.pth')
    # predict on all test data
    return predictions(model, val_dataset, ref_dataset)

def predictions(model, test_dataset, ref_dataset):
    model.eval()
    model.to(DEVICE)
    predictions = []
    ref_batches = []
    for i in np.unique(ref_dataset.target):
        ref_batches.append(ref_dataset.get_batchlbl(i))
    # print(len(ref_batches))
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids in tqdm(test_dataset, desc="Predicting on test data"):
            outputs = []
            input_ids, attention_mask, token_type_ids = input_ids.unsqueeze(0).to(DEVICE), attention_mask.unsqueeze(0).to(DEVICE), token_type_ids.unsqueeze(0).to(DEVICE)
            for ref in ref_batches:
                ref_input_ids, ref_attention_mask, ref_token_type_ids, _ = ref
                ref_input_ids, ref_attention_mask, ref_token_type_ids = ref_input_ids.to(DEVICE), ref_attention_mask.to(DEVICE), ref_token_type_ids.to(DEVICE)
                batch_test = (torch.cat([input_ids]*ref_input_ids.size(0)), torch.cat([attention_mask]*ref_attention_mask.size(0)), torch.cat([token_type_ids]*ref_token_type_ids.size(0)))
                batch_ref = (ref_input_ids, ref_attention_mask, ref_token_type_ids)
                # print(batch_test[0].shape, batch_ref[0].shape)
                output = model(batch_test, batch_ref)
                # pooling output
                output = torch.mean(output, dim=0)
                outputs.append(output)
                # print(output.shape)
                # the predicted label is the index of the reference batch where we found the most similar sentence
            # print(torch.argmax(torch.cat(outputs)).unsqueeze(0))
            predictions.append(torch.argmax(torch.cat(outputs)).unsqueeze(0))

    return torch.cat(predictions).cpu().numpy()


if __name__=="__main__":
    import pandas as pd

    train_dataset = SiameseDataset('./data/augmented_semi.json')
    # additional_dataset = AdditionalDataset('./data/labelled_newscatcher_dataset.csv')
    # train_dataset.extend(additional_dataset)
    labels = train_dataset.labels
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_dataset = TestDataset('./data/test_shuffle.txt')
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    ref_dataset = TrainDataset('./data/augmented.json')
    model = SiameseBert(freeze_bert=True)
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    lin_optimizer = Adam(model.fc.parameters(), lr=5e-3)
    bert_optimizer = Adam(model.bert.parameters(), lr=1e-5)
    # test_preds = train(model, train_dataloader, val_dataset, ref_dataset, criterion, 1, lin_optimizer, bert_optimizer)
    model.load_state_dict(torch.load('./model_weights/sentencebert_augmented_semi.pth'))
    test_preds = predictions(model, val_dataset, ref_dataset)
    test_df = pd.DataFrame()
    test_df["pred"] = test_preds
    test_df['Label'] = test_df['pred'].apply(lambda x: labels[x])
    test_df["ID"] = test_df.index
    test_df.drop(columns=["pred"], inplace=True)
    test_df.to_csv('./data/sentencebert_augmented_semi.csv', index=False)
