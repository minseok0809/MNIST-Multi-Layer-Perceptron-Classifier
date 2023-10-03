import os
import time
import glob
import json
import random
import regex as re
import pandas as pd
from tqdm import tqdm

def sorted_list(path_list):
    
    path_list = sorted(path_list, reverse=False)
    path_list = sorted(path_list, key=len)
    
    return path_list


def json_file_path_list(path_list):
    
    file_path  = [glob.glob(i, recursive = True) for i in path_list][0]
    file_path = sorted_list(file_path)
    
    return file_path


def train_valid_json_file_path_list(path_list):

    train_file_path, valid_file_path = [glob.glob(i, recursive = True) if 'rain' in i
                                        else glob.glob(i, recursive = True)
                                        for i in path_list]

    train_file_path = sorted_list(train_file_path)
    valid_file_path = sorted_list(valid_file_path)
        
    return train_file_path, valid_file_path


def xlsx_file_path_list(file_path, folder_corpus_type_path):
    
    xlsx_file_path = [folder_corpus_type_path + str(i) + ".xlsx"
                                for i in range((len(file_path) // 1000) + 1 )]
        
    return xlsx_file_path


def divide_clickbait_or_nonclibait_json_file_path_list(criterion, json_file_list):

    clickbait_path, nonclickbait_path = [], []
    for x in json_file_list:
        (clickbait_path, nonclickbait_path)[criterion in x].append(x)

    return clickbait_path, nonclickbait_path

def make_train_valid_json_xlsx_file_path_list(json_path_list, xlsx_path_list):

    train_json_file_path, valid_json_file_path = train_valid_json_file_path_list(json_path_list)
    train_clickbait_json_file_path, train_nonclickbait_json_file_path = divide_clickbait_or_nonclibait_json_file_path_list("Non", train_json_file_path)
    valid_clickbait_json_file_path, valid_nonclickbait_json_file_path = divide_clickbait_or_nonclibait_json_file_path_list("Non", valid_json_file_path)
        
    the_number_of_train_clickbait_json_file  = len(train_clickbait_json_file_path)
    the_number_of_valid_clickbait_json_file  = len(valid_clickbait_json_file_path)
    the_number_of_clickbait_json_file = the_number_of_train_clickbait_json_file + the_number_of_valid_clickbait_json_file
    print("The number of train clickbait json file:", the_number_of_train_clickbait_json_file)
    print("The number of valid clickbait json file:", the_number_of_valid_clickbait_json_file)
    print("The number of clickbait json file:", the_number_of_clickbait_json_file)

    the_number_of_train_nonclickbait_json_file  = len(train_nonclickbait_json_file_path)
    the_number_of_valid_nonclickbait_json_file  = len(valid_nonclickbait_json_file_path)
    the_number_of_nonclickbait_json_file = the_number_of_train_nonclickbait_json_file + the_number_of_valid_nonclickbait_json_file
    print()
    print("The number of train nonclickbait json file:", the_number_of_train_nonclickbait_json_file)
    print("The number of valid nonclickbait json file:", the_number_of_valid_nonclickbait_json_file)
    print("The number of nonclickbait json file:", the_number_of_nonclickbait_json_file)

    the_number_of_train_json_file  = len(train_json_file_path)
    the_number_of_valid_json_file  = len(valid_json_file_path)
    the_number_of_json_file = the_number_of_train_json_file + the_number_of_valid_json_file
    print()
    print("The number of train json file:", the_number_of_train_json_file)
    print("The number of valid json file:", the_number_of_valid_json_file)
    print("The number of json file:", the_number_of_json_file)
    
    train_clickbait_xlsx_file_path = xlsx_file_path_list(train_clickbait_json_file_path, xlsx_path_list[0])
    valid_clickbait_xlsx_file_path = xlsx_file_path_list(valid_clickbait_json_file_path, xlsx_path_list[1])
    train_nonclickbait_xlsx_file_path = xlsx_file_path_list(train_nonclickbait_json_file_path, xlsx_path_list[2])
    valid_nonclickbait_xlsx_file_path = xlsx_file_path_list(valid_nonclickbait_json_file_path, xlsx_path_list[3])

    return train_clickbait_json_file_path, train_nonclickbait_json_file_path, \
    valid_clickbait_json_file_path, valid_nonclickbait_json_file_path, \
    train_clickbait_xlsx_file_path, train_nonclickbait_xlsx_file_path, \
    valid_clickbait_xlsx_file_path, valid_nonclickbait_xlsx_file_path


def make_source(json_sample):
    
    title = json_sample['sourceDataInfo']['newsTitle']
    content = json_sample['sourceDataInfo']['newsContent']
    source = title + "\n" +  content
    
    return source

def preprocess_source(source):
    
    source = re.sub(r"\[.*?\]|\{.*?\}|\(.*?\)", "", source)
    source = source.replace("\n", " ")
    source = source.replace("\\\\", "")
    source = source.replace('"', "")
    source = source.replace("'", "")
    source = re.sub(r"[^A-Za-z0-9ㄱ-ㅎ가-힣一-鿕㐀-䶵豈-龎()+-.,]", "", source)

    return source


def write_jsontext_to_xlsx_file_with_batch_size(source_file_list, xlsx_file_path_list, batch_size):

    progress_length = len(source_file_list) // batch_size
    print("[Size]")
    print("The number of xlsx file: " + str(progress_length))
    print("\n[Order]")
    source_list = []
    pbar = tqdm(range(progress_length))
    num = -1

    for i in range(len(source_file_list)):

        source_file = source_file_list[i]
            
        with open(source_file, 'r', encoding='utf-8') as one_json_file:
            one_json_sample = json.load(one_json_file)
            
        source = make_source(one_json_sample)
        # source = preprocess_source(source)
       
        if len(source_list) >= batch_size:
            num += 1
            if 'Non' in source_file:
                label_list = [0] * len(source_list)

            elif 'Non' not in source_file:
                label_list = [1] * len(source_list)
            
            source_df = pd.DataFrame({'Text':source_list, 'Label':label_list})
            source_df_path = xlsx_file_path_list[num]
            source_df.to_excel(source_df_path, index=False)

            pbar.n += 1
            pbar.refresh()
            time.sleep(0.01)  

            source_list = []
                
        elif i == (len(source_file_list) -1): 
            source_list.append(source)
            num += 1
            if 'Non' in source_file:
                label_list = [0] * len(source_list)

            elif 'Non' not in source_file:
                label_list = [1] * len(source_list)
            
            source_df = pd.DataFrame({'Text':source_list, 'Label':label_list})
            source_df_path = xlsx_file_path_list[num]
            source_df.to_excel(source_df_path, index=False)

            pbar.n += 1
            pbar.refresh()
            time.sleep(0.01)
                        
        source_list.append(source)   
    pbar.close()      


def write_xlsxtext_to_list_merge_file(xlsx_folder, dataset_folder):

    xlsx_path = glob.glob(xlsx_folder)
    train_xlsx_path = [train_xlsx for train_xlsx in xlsx_path if 'train' in train_xlsx ]
    valid_xlsx_path = [valid_xlsx for valid_xlsx in xlsx_path if 'valid' in valid_xlsx ]

    train_list = []
    valid_list = []

    for load_path in train_xlsx_path:
        train_xlsx = pd.read_excel(load_path, engine='openpyxl')  
        texts = train_xlsx['Text']
        labels = train_xlsx['Label']
        for text, label in zip(texts, labels):
            text_label = [text, label]
            train_list.append(text_label)

    for load_path in valid_xlsx_path:
        valid_xlsx = pd.read_excel(load_path, engine='openpyxl')  
        texts = valid_xlsx['Text']
        labels = valid_xlsx['Label']
        for text, label in zip(texts, labels):
            text_label = [text, label]
            valid_list.append(text_label)

    random.shuffle(train_list)
    random.shuffle(valid_list)

    train_test_split = int(len(train_list) * 0.99)
    train_copy_list = train_list.copy()
    train_list = train_copy_list[:train_test_split]
    test_list = train_copy_list[train_test_split:]

    train_text = [train[0] for train in train_list]
    train_label = [train[1] for train in train_list]

    valid_text = [valid[0] for valid in valid_list]
    valid_label = [valid[1] for valid in valid_list]

    test_text = [test[0] for test in test_list]
    test_label = [test[1] for test in test_list]

    train_df = pd.DataFrame({'Text':train_text, 'Label':train_label})
    train_df_path = dataset_folder + "train_dataset.xlsx"
    train_df.to_excel(train_df_path, index=False)

    valid_df = pd.DataFrame({'Text':valid_text, 'Label':valid_label})
    valid_df_path = dataset_folder + "valid_dataset.xlsx"
    valid_df.to_excel(valid_df_path, index=False)

    test_df = pd.DataFrame({'Text':test_text, 'Label':test_label})
    test_df_path = dataset_folder + "test_dataset.xlsx"
    test_df.to_excel(test_df_path, index=False)

    print("The number of train text:", len(train_df))
    print("The number of valid text:", len(valid_df))
    print("The number of test text:", len(test_df))

    return train_list, valid_list, test_list