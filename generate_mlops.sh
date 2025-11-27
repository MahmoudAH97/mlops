#!/bin/bash

echo "Creating folder structure..."
mkdir -p environment
mkdir -p components/dataprep
mkdir -p components/split
mkdir -p components/training
mkdir -p pipelines
mkdir -p inference
mkdir -p .github/workflows

# ----------------------------
# ENVIRONMENT FILES
# ----------------------------

cat > environment/compute.yaml << 'EOF'
$schema: https://azuremlschemas.azureedge.net/latest/computeInstance.schema.json

name: cpu-cluster
type: amlcompute
size: Standard_DS2_v2
min_instances: 0
max_instances: 1
idle_time_before_scale_down: 120
EOF

cat > environment/pillow-env.yaml << 'EOF'
$schema: https://azuremlschemas.azureedge.net/latest/environment.schema.json
name: pillow-env
version: 1
image: mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest
conda_file:
  channels: [conda-forge, defaults]
  dependencies:
    - python=3.10
    - pip
    - pip:
      - pillow
EOF

cat > environment/tensorflow-env.yaml << 'EOF'
$schema: https://azuremlschemas.azureedge.net/latest/environment.schema.json
name: tensorflow-env
version: 1
image: mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest
conda_file:
  channels: [conda-forge, defaults]
  dependencies:
    - python=3.10
    - pip
    - pip:
      - tensorflow==2.13.0
      - pillow
      - numpy
      - matplotlib
EOF

# ----------------------------
# DATAPREP COMPONENT
# ----------------------------

cat > components/dataprep/dataprep.yaml << 'EOF'
$schema: https://azuremlschemas.azureedge.net/latest/commandComponent.schema.json
name: dataprep
version: 1
display_name: Data Preparation Component
type: command

inputs:
  data_folder:
    type: uri_folder

outputs:
  output_folder:
    type: uri_folder

environment: azureml:pillow-env:1

command: >-
  python dataprep.py 
  --data_folder ${{inputs.data_folder}} 
  --output_folder ${{outputs.output_folder}}
EOF

cat > components/dataprep/dataprep.py << 'EOF'
import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder")
parser.add_argument("--output_folder")
args = parser.parse_args()

os.makedirs(args.output_folder, exist_ok=True)
shutil.copytree(args.data_folder, args.output_folder, dirs_exist_ok=True)

print("Data prep completed. Files copied.")
EOF

# ----------------------------
# SPLIT COMPONENT
# ----------------------------

cat > components/split/ssplit.yaml << 'EOF'
$schema: https://azuremlschemas.azureedge.net/latest/commandComponent.schema.json
name: split
version: 1
display_name: Train/Test Split Component
type: command

inputs:
  input_folder:
    type: uri_folder
  split_ratio:
    type: number

outputs:
  train_folder:
    type: uri_folder
  test_folder:
    type: uri_folder

environment: azureml:pillow-env:1

command: >-
  python split.py
  --input_folder ${{inputs.input_folder}} 
  --train_folder ${{outputs.train_folder}}
  --test_folder ${{outputs.test_folder}}
  --ratio ${{inputs.split_ratio}}
EOF

cat > components/split/split.py << 'EOF'
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
EOF

# ----------------------------
# TRAINING COMPONENT
# ----------------------------

cat > components/training/training.yaml << 'EOF'
$schema: https://azuremlschemas.azureedge.net/latest/commandComponent.schema.json
name: training
version: 1
display_name: CNN Training Component
type: command

inputs:
  train_folder:
    type: uri_folder
  test_folder:
    type: uri_folder
  epochs:
    type: number

outputs:
  model_output:
    type: uri_folder

environment: azureml:tensorflow-env:1

command: >-
  python training.py
  --train_folder ${{inputs.train_folder}}
  --test_folder ${{inputs.test_folder}}
  --epochs ${{inputs.epochs}}
  --model_output ${{outputs.model_output}}
EOF

cat > components/training/training.py << 'EOF'
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
EOF

# ----------------------------
# PIPELINE
# ----------------------------

cat > pipelines/animals-pipeline.yaml << 'EOF'
$schema: https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json

type: pipeline
experiment_name: animal-exp
display_name: animals-pipeline

inputs:
  split_ratio: 0.8
  epochs: 5

outputs:
  trained_model:
    mode: upload

settings:
  default_compute: azureml:cpu-cluster

jobs:

  prep:
    type: command
    component: ../components/dataprep/dataprep.yaml
    inputs:
      data_folder:
        type: uri_folder
        path: azureml:animal-dataset:1
    outputs:
      output_folder: prep_output

  split:
    type: command
    component: ../components/split/ssplit.yaml
    inputs:
      input_folder: ${{parent.jobs.prep.outputs.output_folder}}
      split_ratio: ${{parent.inputs.split_ratio}}
    outputs:
      train_folder: train_data
      test_folder: test_data

  train:
    type: command
    component: ../components/training/training.yaml
    inputs:
      train_folder: ${{parent.jobs.split.outputs.train_folder}}
      test_folder: ${{parent.jobs.split.outputs.test_folder}}
      epochs: ${{parent.inputs.epochs}}
    outputs:
      model_output: ${{parent.outputs.trained_model}}
EOF

# ----------------------------
# INFERENCE API
# ----------------------------

cat > inference/app.py << 'EOF'
from fastapi import FastAPI, UploadFile
import tensorflow as tf
from PIL import Image
import io

app = FastAPI()

model = tf.keras.models.load_model("model.h5")
class_names = ["cats", "dogs", "panda"]

@app.post("/predict")
async def predict(file: UploadFile):
    img = Image.open(io.BytesIO(await file.read()))
    img = img.resize((128, 128))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = tf.expand_dims(x, 0) / 255.0

    preds = model.predict(x)
    class_id = preds.argmax()

    return {"class": class_names[class_id], "confidence": float(preds[0][class_id])}
EOF

cat > inference/Dockerfile << 'EOF'
FROM python:3.10

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

cat > inference/requirements.txt << 'EOF'
fastapi
uvicorn
tensorflow==2.13.0
pillow
EOF

# ----------------------------
# GITHUB ACTION WORKFLOWS
# ----------------------------

cat > .github/workflows/mlops-train.yml << 'EOF'
# Training pipeline workflow
name: MLOps Train Pipeline

on:
  workflow_dispatch:
  push:
    branches: [ "main" ]

jobs:
  train:
    runs-on: ubuntu-latest

    steps:
    - name: Check out repo
      uses: actions/checkout@v4

    - name: Azure Login
      uses: azure/login@v2
      with:
        creds: ${{ secrets.AZURE_CREDENTIALS }}

    - name: Run Azure ML Pipeline
      uses: azure/CLI@v2
      with:
        inlineScript: |

