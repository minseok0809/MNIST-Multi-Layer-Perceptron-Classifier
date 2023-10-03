import cv2
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import interpolation


def main():

    parer = argparse.ArgumentParser()
    parer.add_argument('--load_model_path', type=str, default='model/mnist_model.sav')
    parer.add_argument('--image_path', type=str, default='./diy/original/')
    parer.add_argument('--save_plot_path', type=str, default='./diy/plot/')
    parer.add_argument('--log_path', type=str, default='log/diy_inference_log.xlsx')
    parer.add_argument('--label', type=list, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    args = parer.parse_args('')       

    load_model_path = args.load_model_path
    
    def diy_mnist_image_inference(loaded_model):
        
        img_paths = glob.glob(args.image_path + "*.png")
        sum = 0

        labels = []
        predictions = []
        results = []

        plt.figure(figsize=(11, 4))
        for idx, img_path in enumerate(img_paths):
            plt.subplot(2, 5, idx+1)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_LINEAR)
            img_resized = cv2.bitwise_not(img_resized)
            alpha = 50.0
            img_resized = np.clip((1+alpha)*img_resized - 128*alpha, 0, 255).astype(np.uint8)
            img_reshape = img_resized.reshape(-1, 784) / 255
            # save_path = img_path.replace("original", "reformat")
            
            labels.append(idx)

            prediction = loaded_model.predict(img_reshape)[0]
            predictions.append(prediction)

            result = str(prediction == idx)
            results.append(result)

            if prediction == idx:
                sum += 1
            
            plt.imshow(img_resized, cmap='gray')
            plt.title(f'prediction: {prediction}')
            plt.axis('off')
        plt.savefig(args.save_plot_path, facecolor='#eeeeee', edgecolor='black', format='png', bbox_inches='tight')
        plt.show()
        num_samples = len(img_paths)
        accuracy = sum / len(img_paths)

        log_df = pd.DataFrame({"Label":labels, "Prediction":predictions, "Result":results})
        log_df.to_excel(args.log_path, index=False)

        return num_samples, accuracy

    loaded_model = pickle.load(open(load_model_path, "rb"))
    loaded_model

    num_samples, accuracy = diy_mnist_image_inference(loaded_model)
    print("\n")
    print('The number of samples: {}'.format(num_samples))
    print('Accuracy: {:3.2f} %'.format(accuracy*100))

if __name__ == '__main__':
    main() 