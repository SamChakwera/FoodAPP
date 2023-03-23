import streamlit as st
import openai
import requests
import json
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

# Set up API keys
openai.api_key = "<your_openai_api_key>"
dalle2_api_key = "<your_dalle2_api_key>"

# Load custom model
extractor = AutoFeatureExtractor.from_pretrained("stchakman/Fridge_Items_Model")
model = AutoModelForImageClassification.from_pretrained("stchakman/Fridge_Items_Model")

# Helper functions
def get_dishes_from_gpt3(ingredients):
    prompt = f"Given the ingredients: {', '.join(ingredients)}, suggest three dishes and their preparation methods."

    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=prompt,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.5,
    )

    return response.choices[0].text.strip()

def generate_images(prompt, num_images=3):
    headers = {
        "Authorization": f"Bearer {dalle2_api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "image-alpha-001",
        "prompt": prompt,
        "num_images": num_images,
    }

    response = requests.post("https://api.openai.com/v1/images/generations", headers=headers, data=json.dumps(data))
    return [image["url"] for image in response.json()["data"]]

def extract_ingredients(image):
    inputs = extractor(image, return_tensors="pt")
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=1).tolist()
    return predictions  # Replace this line with code to map predictions to ingredient names

# Streamlit app
st.title("Fridge Ingredient Recipe Generator")

uploaded_file = st.file_uploader("Upload a photo of your fridge ingredients")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    ingredients = extract_ingredients(image)

    if ingredients:
        dish_text = get_dishes_from_gpt3(ingredients)
        dish_list = dish_text.split("\n")

        dish_images = []
        for dish in dish_list:
            dish_title = dish.split(":")[0].strip()
            dish_images.extend(generate_images(dish_title))

        selected_image = st.select_image(dish_images, width=100, height=100, columns=3)

        if selected_image:
            selected_dish = dish_images.index(selected_image) % 3
            st.markdown(f"**Selected Dish:** {dish_list[selected_dish].split(':')[0].strip()}")
            st.markdown(f"**Preparation:** {dish_list[selected_dish].split(':')[1].strip()}")
