import argparse
import json
import tensorflow as tf
import numpy as np
from tensorflow import keras
from PIL import Image

def load_image(path):
    img = Image.open(path).resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

def predict(image_path, model_path, top_k, class_names_path):
    model = keras.models.load_model(model_path, compile=False)
    img = load_image(image_path)
    preds = model.predict(img)[0]

    top_indices = preds.argsort()[-top_k:][::-1]
    top_probs = preds[top_indices]

    if class_names_path:
        with open(class_names_path, "r") as f:
            class_names = json.load(f)
        top_labels = [class_names[str(i + 1)] for i in top_indices]
    else:
        top_labels = [str(i) for i in top_indices]

    return top_probs, top_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    parser.add_argument("model_path", type=str)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--category_names", type=str, default=None)
    args = parser.parse_args()

    probs, labels = predict(
        args.image_path,
        args.model_path,
        args.top_k,
        args.category_names
    )

    for p, name in zip(probs, labels):
        print(f"{name}: {p:.4f}")

if __name__ == "__main__":
    main()
#python predict.py ./test_images/hard-leaved_pocket_orchid.jpg ./my_model