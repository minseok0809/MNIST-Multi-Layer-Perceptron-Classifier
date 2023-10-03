
import glob
import random
import argparse
import pandas as pd 

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, default='./data/aihub_clickbait/dataset/')
    parser.add_argument('--label', type=list, default=[0, 1])
    args = parser.parse_args('')   

    def write_xlsxtext_to_test_list(dataset_folder):

        train_list = []
        test_list = []

        test_xlsx_path = dataset_folder + "test_dataset.xlsx"
        test_xlsx = pd.read_excel(test_xlsx_path, engine='openpyxl')

        texts = test_xlsx['Text']
        labels = test_xlsx['Label']
        for text, label in zip(texts, labels):
            text_label = [text, label]
            test_list.append(text_label)

        return test_list

    def evaluate(y_test, y_anon):
        sum = 0
        for i, j in zip(y_test, y_anon):
            if i == j:
                sum += 1
        accuracy = sum / len(y_test)
        return accuracy

    def random_baseline(test_dataset, label, dataset_folder):
        y_test = [test[1] for test in test_dataset]
        num = len(y_test)
        y_anon = []

        for i in range(0, num):
            n = random.choice(label)
            y_anon.append(n)

        accuracy = evaluate(y_test, y_anon)
        print('Accuracy: {:3.2f} %'.format(accuracy*100))

        y_anon_xlsx = pd.DataFrame({'Label': y_anon})
        y_anon_xlsx_path = dataset_folder + "test_dataset.xlsx"
        y_anon_xlsx.to_excel(y_anon_xlsx_path, index=False)
    
    test_dataset = write_xlsxtext_to_test_list(args.dataset_folder)
    random_baseline(test_dataset, args.label, args.dataset_folder)

if __name__ == '__main__':
    main()    
