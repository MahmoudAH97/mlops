import os
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--train_folder")
parser.add_argument("--test_folder")
parser.add_argument("--epochs", type=int)
parser.add_argument("--model_output")
args = parser.parse_args()

img_size = (128, 128)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    args.train_folder, image_size=img_size, batch_size=16)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    args.test_folder, image_size=img_size, batch_size=16)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(len(train_ds.class_names), activation="softmax")
])

model.compile(
    optimizer="adam", 
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_ds, validation_data=test_ds, epochs=args.epochs)

os.makedirs(args.model_output, exist_ok=True)
model.save(os.path.join(args.model_output, "model.h5"))

print("Model training completed.")
