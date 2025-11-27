import os
import argparse
from PIL import Image

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    args = parser.parse_args()

    input_dir = args.data_folder
    output_dir = args.output_folder

    os.makedirs(output_dir, exist_ok=True)

    # Copy and validate images
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                src = os.path.join(root, f)

                # keep class folder name
                class_name = os.path.basename(os.path.dirname(src))
                class_out = os.path.join(output_dir, class_name)
                os.makedirs(class_out, exist_ok=True)

                dst = os.path.join(class_out, f)

                # Validate image
                try:
                    img = Image.open(src)
                    img.verify()
                    Image.open(src).save(dst)
                except Exception as e:
                    print(f"Skipping corrupted file: {src} ({e})")

if __name__ == "__main__":
    run()
