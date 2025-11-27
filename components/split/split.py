import os
import argparse
import shutil
import random

parser = argparse.ArgumentParser()
parser.add_argument("--input_folder")
parser.add_argument("--train_folder")
parser.add_argument("--test_folder")
parser.add_argument("--ratio", type=float)
args = parser.parse_args()

random.seed(42)

os.makedirs(args.train_folder, exist_ok=True)
os.makedirs(args.test_folder, exist_ok=True)

classes = os.listdir(args.input_folder)

for cls in classes:
    src = os.path.join(args.input_folder, cls)
    files = os.listdir(src)
    random.shuffle(files)

    cutoff = int(len(files) * args.ratio)
    train_files = files[:cutoff]
    test_files = files[cutoff:]

    os.makedirs(os.path.join(args.train_folder, cls), exist_ok=True)
    os.makedirs(os.path.join(args.test_folder, cls), exist_ok=True)

    for f in train_files:
        shutil.copy(os.path.join(src, f), os.path.join(args.train_folder, cls))

    for f in test_files:
        shutil.copy(os.path.join(src, f), os.path.join(args.test_folder, cls))

print("Train/test split complete.")
