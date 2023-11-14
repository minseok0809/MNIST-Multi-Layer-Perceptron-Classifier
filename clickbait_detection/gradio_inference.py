import gradio as gr
import pandas as pd 
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main(text):
    
    class ClickbaitDetectionDataset(Dataset):
    
        def __init__(self, dataset):
            self.dataset = dataset
            self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
        
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
    model = AutoModelForSequenceClassification.from_pretrained("klue/roberta-base").to(device)
    checkpoint = torch.load("model/clickbait_classifcation_model_5.bin")
    model.load_state_dict(checkpoint)
    model.eval()
    
    test_data = pd.DataFrame({"Text":[text]})
    test_dataset = ClickbaitDetectionDataset(test_data)
    
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    with torch.no_grad():   
        for input_ids_batch, attention_masks_batch in test_loader:
            # optimizer.zero_grad()
            
            y_batch = y_batch.to(device)
            y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
            _, predicted = torch.max(y_pred, 1)

            for prediction in predicted.tolist():
                if prediction == 0:
                    classfication_result = "It's Not Clickbait Article"
                elif prediction == 1:
                    classfication_result =  "It's Clickbait Article"

    return classfication_result

        
if __name__ == '__main__':
    classfication_resul = main()    
    demo = gr.Interface(fn=main, inputs="text", outputs="text")
    demo.launch()  
    # demo.launch( share = True , debug = True)  