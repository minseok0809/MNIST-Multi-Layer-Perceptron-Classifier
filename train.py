import pickle
import argparse
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report


def main():

    warnings.filterwarnings( 'ignore' )

    parer = argparse.ArgumentParser()
    parer.add_argument('--train_dataset', type=str, default='data/train.csv')
    parer.add_argument('--save_model_path', type=str, default='data/mnist_model.sav')
    parer.add_argument('--log_path', type=str, default='log/training_log.xlsx')
    parer.add_argument('--label', type=list, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    args = parer.parse_args('')   

    X_train = pd.read_csv(args.train_dataset)
    X_train = X_train.rename(columns={'5':'label'})
    y_train = X_train['label'] 
    x_train = X_train.drop(['label'], axis=1)

    train_x = (x_train.values) / 255
    train_y = y_train.values.flatten() 

    x_train, x_dev, y_train, y_dev = train_test_split(train_x, train_y, random_state=42)

    model = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(1000,),  random_state=1) 
    trained_model = model.fit(x_train, y_train)

    pickle.dump(trained_model, open(args.save_model_path, 'wb'))

    y_train = pd.DataFrame({"Label": X_train['label']})

    y_train['Prediction'] = trained_model.predict(train_x)
    y_train['Result'] = y_train.apply(lambda row: row['Prediction']==row['Label'], axis=1)
    y_train.to_excel(args.log_path, index=False)

    print('Training Accuracy: {:3.2f} %'.format(trained_model.score(train_x, train_y) * 100))
    y_test_pred = trained_model.predict(train_x)

if __name__ == '__main__':
    main()    
