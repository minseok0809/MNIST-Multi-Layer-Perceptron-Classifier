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

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='klue/roberta-base')
    parser.add_argument('--data_path', type=str, default='./data/aihub_clickbait_detection/dataset/')
    parser.add_argument('--test_dataset', type=str, default='test_dataset.xlsx')
    parser.add_argument('--load_model_path', type=str, default='./model')
    parser.add_argument('--log_path', type=str, default='./log/clickabait_detection_evaluation_log.xlsx')
    parser.add_argument('--inference_log_path', type=str, default='./log/clickabait_detection_inference_log.xlsx')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
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
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name).to(device)
    save_model_name = (((args.model_name).split("/"))[1]).replace("-", "_")
    checkpoint = torch.load(os.path.join(args.load_model_path, save_model_name + "_clickbait_"  + str(args.epoch) + ".bin"))
    model.load_state_dict(checkpoint)
    model.eval()
    
    test_data_path = args.data_path + args.test_dataset
    test_data = pd.read_excel(test_data_path, engine='openpyxl')
    # test_data = test_data.iloc[:500,:]

    test_dataset = ClickbaitDetectionDataset(test_data)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    inference_index = [num for num in range(1, len(test_data)+1)]
    
    inference_log_df = pd.DataFrame({'index':inference_index,
                                     'Label':test_data['Label']})  
    prediction = []
    test_time = []
    test_accuracy = []
    
    with torch.no_grad():
        correct = 0
        total = 0
        batches = 0
        num = 0
        
        with tqdm(test_loader) as pbar: 
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

        inference_log_df['Prediction'] = prediction
        inference_log_df.to_excel(args.inference_log_path, index=False)
        
        y_pred_xlsx_path = args.inference_log_path
        y_pred_xlsx_split = y_pred_xlsx_path.split("/")
        
        json_path = ""
        for i in y_pred_xlsx_split:
            if i != '.' and 'clickabait' not in i:
                json_path += (i + "/")
        
        real_result_json_path = json_path + "real_result.json"
        predict_result_json_path = json_path + "predict_result.json"

        y_test_df = inference_log_df[['index','Label']]      
        y_test_df.columns = ['ID','Class']
        y_pred_df = inference_log_df[['index','Prediction']]
        y_pred_df.columns = ['ID','Class']

        y_test_df.to_json(real_result_json_path, orient = 'split', indent = 4, index=False)
        y_pred_df.to_json(predict_result_json_path, orient = 'split', indent = 4, index=False)

        if len(elapsed_str) == 5:
            elapsed_str = "00:" + elapsed_str
        elapsed_str = str(datetime.datetime.strptime(elapsed_str, '%H:%M:%S').time())    
        
        pbar.close()  
        test_time.append(elapsed_str) 
        accuracy = round((correct.float() / total).item(), 4)
        test_accuracy.append(accuracy)
        print("Test Time",  elapsed_str, "  ", "Test Accuracy:", accuracy)  


    model_log_df = pd.DataFrame({'Test Time':test_time, 
                                            'Test Accuracy':test_accuracy})    
    model_log_df.to_excel(args.log_path, index=False)
        
if __name__ == '__main__':
    main()    
