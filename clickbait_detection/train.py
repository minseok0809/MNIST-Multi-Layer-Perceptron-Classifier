import os
import datetime
import regex as re
import argparse
import pandas as pd 
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, RobertaForSequenceClassification, AdamW


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='klue/roberta-base')
    parser.add_argument('--data_path', type=str, default='./data/aihub_clickbait_detection/dataset/')
    parser.add_argument('--train_dataset', type=str, default='train_dataset.xlsx')
    parser.add_argument('--valid_dataset', type=str, default='valid_dataset.xlsx')
    parser.add_argument('--save_model_path', type=str, default='./model')
    parser.add_argument('--log_path', type=str, default='./log/clickabait_detection_training_log.xlsx')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()   
    
    class ClickbaitDetectionDataset(Dataset):
    
        def __init__(self, dataset):
            self.dataset = dataset
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            row = self.dataset.iloc[idx, 0:2].values
            text = row[0]
            y = row[1]

            inputs = self.tokenizer(
                text, 
                return_tensors='pt',
                truncation=True,
                max_length=256,
                pad_to_max_length=True,
                add_special_tokens=True
                )
            
            input_ids = inputs['input_ids'][0]
            attention_mask = inputs['attention_mask'][0]

            return input_ids, attention_mask, y    
            
    device = torch.device("cuda")
    model = RobertaForSequenceClassification.from_pretrained(args.model_name).to(device)

    train_data_path = args.data_path + args.train_dataset
    valid_data_path = args.data_path + args.valid_dataset
    train_data = pd.read_excel(train_data_path, engine='openpyxl')
    valid_data = pd.read_excel(valid_data_path, engine='openpyxl')
    
    train_dataset = ClickbaitDetectionDataset(train_data)
    valid_dataset = ClickbaitDetectionDataset(valid_data)
    
    optimizer = AdamW(model.parameters(), lr=1e-5)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

    train_time = []
    train_loss = []
    train_accuracy = []

    for i in range(args.epoch):
        total_loss = 0.0
        correct = 0
        total = 0
        batches = 0

        model.train()
        
        with tqdm(train_loader) as pbar:
            pbar.set_description("Epoch " + str(i + 1))        
            for input_ids_batch, attention_masks_batch, y_batch in pbar:
                optimizer.zero_grad()
                y_batch = y_batch.to(device)
                y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
                one_loss = F.cross_entropy(y_pred, y_batch)
                one_loss.backward()
                optimizer.step()

                total_loss += one_loss.item()

                _, predicted = torch.max(y_pred, 1)
                correct += (predicted == y_batch).sum()
                total += len(y_batch)

                batches += 1
                # if batches % 100 == 0:
                # print("Batch Loss:", total_loss, "Accuracy:", correct.float() / total)

                elapsed = pbar.format_dict['elapsed']
                elapsed_str = pbar.format_interval(elapsed)
                
        save_model_name = (((args.model_name).split("/"))[1]).replace("-", "_")
        torch.save(model.state_dict(), os.path.join(args.save_model_path, save_model_name + "_clickbait_" + str(i+1) + ".bin"))

        if len(elapsed_str) == 5:
            elapsed_str = "00:" + elapsed_str
        elapsed_str = str(datetime.datetime.strptime(elapsed_str, '%H:%M:%S').time())    
        
        pbar.close()  
        train_time.append(elapsed_str)
        total_loss = round(total_loss, 4)                             
        train_loss.append(total_loss)
        accuracy = round((correct.float() / total).item(), 4)
        train_accuracy.append(accuracy)
        print("Train Time",  elapsed_str, "  ", "Train Loss:", total_loss,  "  ",  "Train Accuracy:", accuracy)    

    valid_time = []
    valid_loss = []
    valid_accuracy = []
    
    with torch.no_grad():
        
        for i in range(args.epoch):
            
            total_loss = 0.0
            correct = 0
            total = 0
            batches = 0
            model = RobertaForSequenceClassification.from_pretrained(args.model_name).to(device)
            checkpoint = torch.load(os.path.join(args.save_model_path, save_model_name + "_clickbait_" + str(i+1) + ".bin"))
            model.load_state_dict(checkpoint)
            model.eval()
            
            with tqdm(valid_loader) as pbar:
                pbar.set_description("Epoch " + str(i + 1))   
                for input_ids_batch, attention_masks_batch, y_batch in pbar:
                    # optimizer.zero_grad()
                    y_batch = y_batch.to(device)
                    y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
                    one_loss = F.cross_entropy(y_pred, y_batch)
                    # one_loss.backward()
                    # optimizer.step()

                    total_loss += one_loss.item()

                    _, predicted = torch.max(y_pred, 1)
                    correct += (predicted == y_batch).sum()
                    total += len(y_batch)

                    batches += 1
                    # if batches % 100 == 0:
                    # print("Batch Loss:", total_loss, "Accuracy:", correct.float() / total)
                elapsed = pbar.format_dict['elapsed']
                elapsed_str = pbar.format_interval(elapsed)
                
            if len(elapsed_str) == 5:
                elapsed_str = "00:" + elapsed_str
            elapsed_str = str(datetime.datetime.strptime(elapsed_str, '%H:%M:%S').time())    
            
            pbar.close()  
            valid_time.append(elapsed_str) 
            total_loss = round(total_loss, 4)   
            valid_loss.append(total_loss)
            accuracy = round((correct.float() / total).item(), 4)
            valid_accuracy.append(accuracy)
            print("Valid Time",  elapsed_str, "  ", "Valid Loss:", total_loss, "  ", "Valid Accuracy:", accuracy)  

    epoch = [num for num in range(1, args.epoch+1)]
    
    model_log_df = pd.DataFrame({'Epoch':epoch, 'Training Time':train_time,  
                                 'Training Loss':train_loss, 'Training Accuracy':train_accuracy,
                                 'Validation Time':valid_time, 
                                 'Validation Loss':valid_loss, 'Validation Accuracy':valid_accuracy})    
    model_log_df.to_excel(args.log_path, index=False)
        
if __name__ == '__main__':
    main()    
