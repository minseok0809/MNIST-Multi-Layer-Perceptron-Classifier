import os
import glob
import datetime
import regex as re
import argparse
import pandas as pd 
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, RobertaForSequenceClassification


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='klue/roberta-base')
    parser.add_argument('--news_article_text_folder', type=str, default='./data/aihub_clickbait_detection/news_article/')
    parser.add_argument('--data_path', type=str, default='./data/aihub_clickbait_detection/dataset/')
    parser.add_argument('--dataset', type=str, default='dataset.xlsx')
    parser.add_argument('--label', '--names-list', nargs='+', default=[])
    parser.add_argument('--load_model_path', type=str, default='./model')
    parser.add_argument('--clickbait_detection_log_path', type=str, default='./log/clickabait_detection_log.df')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()   
    
    """
    clickbait_aritlce =
    [https://tenasia.hankyung.com/tv-drama/article/2023030212854]
    """
    
    data_df = pd.DataFrame({'Text':[0], 'Label':[0]}) 
    news_article_text_files = glob.glob(args.news_article_text_folder + "*.txt")
    news_article_text = ""
    
    for idx, (news_article_text_file, label) in enumerate(zip(news_article_text_files, args.label)):
        with open(news_article_text_file, "r", encoding='utf-8') as f:
            lines = f.read().splitlines() 
            for idx, line in enumerate(lines):
                if idx == 0:
                    news_article_text += line
                elif idx > 0:
                    news_article_text += " " + line    
        data_df.loc[idx] = [news_article_text, label]
        
    data_df = data_df.drop([0], axis = 0)
    data_df = data_df.astype({'Label':int})    
    data_path = args.data_path + args.dataset
    data_df.to_excel(data_path, index=False) 
    
    
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
    
    save_model_name = (((args.model_name).split("/"))[1]).replace("-", "_")
    checkpoint = torch.load(os.path.join(args.load_model_path, save_model_name + "_clickbait_" + str(args.epoch) + ".bin"))
    model.load_state_dict(checkpoint)
    model.eval()
    
    data = pd.read_excel(data_path, engine='openpyxl')
    
    dataset = ClickbaitDetectionDataset(data)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    prediction = []
    

    with torch.no_grad():
        correct = 0
        total = 0
        batches = 0
        num = 0
        
        with tqdm(data_loader) as pbar: 
            for input_ids_batch, attention_masks_batch, y_batch in pbar:
                # optimizer.zero_grad()
                
                y_batch = y_batch.to(device)
                y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]

                _, predicted = torch.max(y_pred, 1)
                correct += (predicted == y_batch).sum()
                total += len(y_batch)
                
                batches += 1
                prediction.append(predicted.item())   
                # if batches % 100 == 0:
                # print("Batch Loss:", total_loss, "Accuracy:", correct.float() / total)
            elapsed = pbar.format_dict['elapsed']
            elapsed_str = pbar.format_interval(elapsed)
        
            if len(elapsed_str) == 5:
                elapsed_str = "00:" + elapsed_str
            elapsed_str = str(datetime.datetime.strptime(elapsed_str, '%H:%M:%S').time())    
            
            pbar.close()  
            accuracy = round((correct.float() / total).item())
            print("Time",  elapsed_str, "  ", "Accuracy:", accuracy)  
        
    data['Prediction'] = [prediction]
    data['Time'] = [elapsed_str]
    data['Accuracy'] = [accuracy]
    data.to_excel(args.clickbait_detection_log_path, index=False)
        
if __name__ == '__main__':
    main()    
