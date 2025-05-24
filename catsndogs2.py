from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import random

DATASET_PATH = 'dataset/'
class_names = sorted(os.listdir(DATASET_PATH))
class_mode = "binary" if len(class_names) == 2 else "categorical"

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File not found at path: {image_path}")
        return

    try:
        img = Image.open(image_path)
        img.verify()
        img = Image.open(image_path)
    except (OSError, IOError):
        print(f"Error: Damaged image - {image_path}")
        return

    model = tf.keras.models.load_model("image_classifier.h5")

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to read image - {image_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)

    prediction = model.predict(img)

    if class_mode == "binary":
        predicted_class = class_names[1 if prediction[0][0] > 0.5 else 0]
    else:
        predicted_class = class_names[tf.argmax(prediction, axis=-1).numpy()[0]]

    print(f"Model predicted: {predicted_class}")

    img_display = Image.open(image_path)
    plt.imshow(img_display)
    plt.title(f"Model concluded: {predicted_class}")
    plt.axis('off')
    plt.show()

def predict_random_image():
    chosen_class = random.choice(class_names)
    class_folder = os.path.join(DATASET_PATH, chosen_class)
    image_files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"No images found in {class_folder}")
        return
    chosen_image = random.choice(image_files)
    image_path = os.path.join(class_folder, chosen_image)
    print(f"Predicting on random image: {image_path}")
    predict_image(image_path)

predict_random_image()
