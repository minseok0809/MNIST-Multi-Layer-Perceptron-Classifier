import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
from keras.utils import to_categorical
from pathlib import Path, PureWindowsPath


def train(label="train.csv", model_save_path="model.keras"):
    metadata_df = pd.read_csv(label, sep=",")
    metadata_df = metadata_df[["wav_file", "species"]]
    metadata_df["wav_file"] = [
        os.path.join(os.path.dirname(label), str(Path(PureWindowsPath(path))))
        for path in metadata_df["wav_file"]
    ]
    metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)
    metadata_df.head(3)

    split = int(len(metadata_df) * 0.90)
    df_train = metadata_df[:split]
    df_val = metadata_df[split:]

    print(f"Size of the training set: {len(df_train)}")
    print(f"Size of the validation set: {len(df_val)}")

    frame_length = 256
    # An integer scalar Tensor. The number of samples to step.
    frame_step = 160
    # An integer scalar Tensor. The size of the FFT to apply.
    # If not provided, uses the smallest power of 2 enclosing frame_length.
    fft_length = 384

    def encode_single_sample(wav_file, label):
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

        # 7. Convert label to categorical case
        return spectrogram, label

    batch_size = 16
    # Define the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (
            df_train["wav_file"].to_numpy(dtype=np.string_),
            to_categorical(df_train["species"].to_numpy(dtype=np.int_), num_classes=6),
        )
    )
    train_dataset = (
        train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # Define the validation dataset
    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (
            df_val["wav_file"].to_numpy(dtype=np.string_),
            to_categorical(df_val["species"].to_numpy(dtype=np.int_), num_classes=6),
        )
    )
    validation_dataset = (
        validation_dataset.map(
            encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
        )
        .padded_batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    def build_model(num_classes):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    64,
                    (3, 3),
                    strides=2,
                    padding="same",
                    input_shape=(None, None, 1),
                    activation="relu",
                ),
                tf.keras.layers.MaxPooling2D(
                    pool_size=(3, 3), strides=2, padding="same"
                ),
                tf.keras.layers.Conv2D(
                    128,
                    (3, 3),
                    strides=1,
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.MaxPooling2D(
                    pool_size=(3, 3), strides=2, padding="same"
                ),
                tf.keras.layers.Conv2D(
                    256,
                    (3, 3),
                    strides=1,
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.MaxPooling2D(
                    pool_size=(3, 3), strides=2, padding="same"
                ),
                tf.keras.layers.Conv2D(
                    512,
                    (3, 3),
                    strides=1,
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(num_classes),
                tf.keras.layers.Softmax(),
            ]
        )
        # Optimizer
        opt = keras.optimizers.Adam(learning_rate=1e-4)
        # Compile the model and return
        model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy())
        return model

    # Get the model
    model = build_model(num_classes=6)
    model.summary(line_length=110)

    # Define the number of epochs.
    epochs = 20
    print()
    # Train the model
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=keras.callbacks.BackupAndRestore("./backups"),
    )
    model.save(model_save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="baseline stt trainer")
    parser.add_argument(
        "--y_train_path",
        type=str,
        default="data/train.csv",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="model.keras",
    )
    args = parser.parse_args()
    train(args.y_train_path, args.model_save_path)
