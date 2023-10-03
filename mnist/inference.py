import pickle
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report


def main():
    
    parer = argparse.ArgumentParser()
    parer.add_argument('--test_dataset', type=str, default='data/t10k.csv')
    parer.add_argument('--load_model_path', type=str, default='data/mnist_model.sav')
    parer.add_argument('--log_path', type=str, default='log/inference_log.xlsx')
    parer.add_argument('--label', type=list, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    args = parer.parse_args('')   

    X_test = pd.read_csv(args.test_dataset)
    X_test = X_test.rename(columns={'7':'label'})
    y_test = X_test['label'] 
    x_test = X_test.drop(['label'], axis=1)

    test_x = (x_test.values) / 255
    test_y = y_test.values.flatten() 

    x_test, x_dev, y_test, y_dev = train_test_split(test_x, test_y, random_state=42)

    loaded_model = pickle.load(open(args.load_model_path, 'rb'))

    y_test= pd.DataFrame({"Label": X_test['label']})
    y_test['Prediction'] = loaded_model.predict(test_y)
    y_test['Result'] = y_test.apply(lambda row: row['Prediction']==row['Label'], axis=1)
    y_test.to_excel(args.log_path, index=False)

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
