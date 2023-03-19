import streamlit as st
import requests
import json
import base64
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import os
# Replace YOUR_API_KEY with your actual OpenAI API key
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_API_URL = "https://api.openai.com/v1/engines/davinci-codex/completions"
DALLE_API_URL = "https://api.openai.com/v1/images/generations"

def load_image_classification_model():
    model = MobileNetV2(weights="imagenet")
    return model

def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

def get_food_items(image):
    model = load_image_classification_model()
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    decoded_preds = decode_predictions(predictions, top=5)[0]

    food_items = []
    for _, item, _ in decoded_preds:
        food_items.append(item)

    return food_items

def ask_gpt3(food_items):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    prompt = f"Given the following ingredients: {', '.join(food_items)}, what dishes can be created?"

    data = {
        "engine": "text-davinci-003",
        "prompt": prompt,
        "max_tokens": 100,
        "n": 1,
        "stop": None,
        "temperature": 0.7,
    }

    response = requests.post(OPENAI_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        suggestion = response.json()["choices"][0]["text"].strip()
    else:
        suggestion = "Error: Unable to get a suggestion from GPT-3."
        st.write(f"Error details: {response.status_code}, {response.text}")  # Print the error details

    return suggestion


def generate_dalle_image(prompt):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "image-alpha-001",
        "prompt": prompt,
        "num_images": 1,
        "size": "256x256",
    }

    response = requests.post(DALLE_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        image_url = response.json()["data"][0]["url"]
    else:
        image_url = None
        st.error("Error: Unable to generate an image using DALL-E 2.")

    return image_url


st.title("Fridge Recipe Suggester")
uploaded_file = st.file_uploader("Upload a picture of your fridge", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded fridge image", use_column_width=True)

    food_items = get_food_items(image)
    st.write(f"Food items identified: {', '.join(food_items)}")

    dish = ask_gpt3(food_items)
    st.write(f"Suggested dish: {dish}")

    image_prompt = f"An image of {dish}"
    encoded_image = generate_dalle_image(image_prompt)
    decoded_image = base64.b64decode(encoded_image)
    dalle_image = Image.open(io.BytesIO(decoded_image))
    st.image(dalle_image, caption=f"{dish} (Generated by DALL-E 2)", use_column_width=True)
