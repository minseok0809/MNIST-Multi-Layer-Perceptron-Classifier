import pickle
import argparse
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report


def main():

    warnings.filterwarnings( 'ignore')
    
    parer = argparse.ArgumentParser()
    parer.add_argument('--test_dataset', type=str, default='data/t10k.csv')
    parer.add_argument('--load_model_path', type=str, default='model/mnist_model.sav')
    parer.add_argument('--log_path', type=str, default='log/inference_log.xlsx')
    parer.add_argument('--label', type=list, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    args = parer.parse_args('')   

    X_test = pd.read_csv(args.test_dataset)
    X_test = X_test.rename(columns={'7':'label'})
    y_test = X_test['label'] 
    x_test = X_test.drop(['label'], axis=1)

    test_x = (x_test.values) / 255
    test_y = y_test.values.flatten() 

    loaded_model = pickle.load(open(args.load_model_path, 'rb'))

    y_test = pd.DataFrame({"Label": X_test['label']})

    y_test['Prediction'] = loaded_model.predict(test_x)
    y_test['Result'] = y_test.apply(lambda row: row['Prediction']==row['Label'], axis=1)
    y_test.to_excel(args.log_path, index=False)

    y_true_csv_path = "log/y_true.csv"
    y_pred_csv_path = "log/y_pred.csv"
    y_true_xlsx_path = "log/y_true.xlsx"
    y_pred_xlsx_path = "log/y_pred.xlsx"
    real_result_json_path = "log/real_result.json"
    predict_result_json_path = "log/predict_result.json"

    y_test = y_test.reset_index(drop = False) 
    y_test_df = y_test['Label']
    y_test_df.columns = ['ID','Class']
    y_pred_df = y_test['Prediction']
    y_pred_df.columns = ['ID','Class']

    # y_test_df.to_csv(y_true_csv_path, index=False)
    # y_pred_df.to_csv(y_pred_csv_path, index=False)

    # y_test_df.to_excel(y_true_xlsx_path, index=False)
    # y_pred_df.to_excel(y_pred_xlsx_path, index=False)

    y_test_df.to_json(real_result_json_path, orient = 'split', indent = 4, index=False)
    y_pred_df.to_json(predict_result_json_path, orient = 'split', indent = 4, index=False)

    def evaluate(y_result):
        sum = 0
        for result in y_result:
            if result == True:
                sum += 1
        accuracy = sum / len(y_result)
        return accuracy

    y_result = y_test['Result']
    accuracy = evaluate(y_result)
    print('Accuracy: {:3.2f} %'.format(accuracy*100))

if __name__ == '__main__':
    main()    
