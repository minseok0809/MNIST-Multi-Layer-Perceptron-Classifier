import numpy as np
import sys
from tensorflow.keras.models import load_model
import pandas as pd
import tensorflow as tf
import os
import argparse

frame_length = 256
# An integer scalar Tensor. The number of samples to step.
frame_step = 160
# An integer scalar Tensor. The size of the FFT to apply.
# If not provided, uses the smallest power of 2 enclosing frame_length.
fft_length = 384


def encode_single_sample(wav_file):
    # 1. Read wav file
    file = tf.io.read_file(wav_file)
    # 2. Decode the wav file
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    # 3. Change type to float
    audio = tf.cast(audio, tf.float32)
    # 4. Get the spectrogram
    spectrogram = tf.signal.stft(
        audio,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
    )
    # 5. We only need the magnitude, which can be derived by applying tf.abs
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    # 6. normalisation
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    return spectrogram


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model inference")
    parser.add_argument("--model_path", default="./model.keras", help="Path to the model checkpoint")
    parser.add_argument("--x_test_path", default="./data/random_test.csv",help="Path to the x_test files")
    parser.add_argument("--y_pred_save_path", default="./pred_test.csv",help="Path to the y_pred file")
    args = parser.parse_args()
    # File paths
    x_test_path = args.x_test_path
    csv_file = args.y_pred_save_path
    model_path = args.model_path

    # Load the model
    model = load_model(model_path)

    df = pd.read_csv(x_test_path)
    predictions = []
    print()
    for index, row in df.iterrows():
        wav_file_path = "./data/" + row["wav_file"]
        spectrogram = encode_single_sample(wav_file_path)
        prediction = model.predict(tf.expand_dims(spectrogram, axis=0))
        predicted_label = tf.argmax(prediction, axis=1).numpy()[0]
        predictions.append(predicted_label)

    # Add the predictions to the csv
    df["species"] = predictions
    df.to_csv(csv_file, index=False)
