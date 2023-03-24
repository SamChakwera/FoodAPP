import streamlit as st
import requests
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from transformers import pipeline
import openai
from io import BytesIO
import os
import tempfile
from diffusers import StableDiffusionPipeline
import torch
import base64

openai.api_key = os.getenv("OPENAI_API_KEY")

# Load models and set up GPT-3 pipeline
extractor = AutoFeatureExtractor.from_pretrained("stchakman/Fridge_Items_Model")
model = AutoModelForImageClassification.from_pretrained("stchakman/Fridge_Items_Model")
#gpt3 = pipeline("text-davinci-003", api_key="your_openai_api_key")

# Map indices to ingredient names
term_variables = { "Apples", "Asparagus", "Avocado", "Bananas", "BBQ sauce", "Beans", "Beef", "Beer", "Berries", "Bison", "Bread", "Broccoli", "Cauliflower", "Celery", "Cheese", "Chicken", "Chocolate", "Citrus fruits", "Clams", "Cold cuts", "Corn", "Cottage cheese", "Crab", "Cream", "Cream cheese", "Cucumbers", "Duck", "Eggs", "Energy drinks", "Fish", "Frozen vegetables", "Frozen meals", "Garlic", "Grapes", "Ground beef", "Ground chicken", "Ham", "Hot sauce", "Hummus", "Ice cream", "Jams", "Jerky", "Kiwi", "Lamb", "Lemons", "Lobster", "Mangoes", "Mayonnaise", "Melons", "Milk", "Mussels", "Mustard", "Nectarines", "Onions", "Oranges", "Peaches", "Peas", "Peppers", "Pineapple", "Pizza", "Plums", "Pork", "Potatoes", "Salad dressings", "Salmon", "Shrimp", "Sour cream", "Soy sauce", "Spinach", "Squash", "Steak", "Sweet potatoes", "Frozen Fruits", "Tilapia", "Tomatoes", "Tuna", "Turkey", "Venison", "Water bottles", "Wine", "Yogurt", "Zucchini" }
ingredient_names = list(term_variables)

classifier = pipeline("image-classification", model="stchakman/Fridge_Items_Model")

def extract_ingredients(uploaded_image):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_image.getvalue())
    temp_file.flush()

    image = Image.open(temp_file.name)
    preds = classifier(temp_file.name)
    ingredients = [pred["label"] for pred in preds]

    temp_file.close()
    os.unlink(temp_file.name)
    return ingredients


def generate_dishes(ingredients, n=3, max_tokens=150, temperature=0.7):
    ingredients_str = ', '.join(ingredients)
    prompt = f"I have {ingredients_str} Please return the name of a dish I can make followed by instructions on how to prepare that dish"

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n
    )

    dishes = [choice.text.strip() for choice in response.choices]
    return dishes

model_id = "runwayml/stable-diffusion-v1-5"
def generate_image(prompt):
    with st.spinner("Generating image..."):
        pipe = StableDiffusionPipeline.from_pretrained(model_id)
    # If you have a GPU available, uncomment the following line
    # pipe = pipe.to("cuda")
        image = pipe(prompt).images[0]
    return image

def get_image_download_link(image, filename, text):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a download="{filename}" href="data:image/jpeg;base64,{img_str}" target="_blank">{text}</a>'
    return href

st.title("Fridge to Dish App")

uploaded_file = st.file_uploader("Upload an image of your ingredients", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    ingredients = extract_ingredients(uploaded_file)
    st.write("Ingredients found:")
    st.write(", ".join(ingredients))
    
    suggested_dishes = generate_dishes(ingredients)

    if len(suggested_dishes) > 0:
        st.write("Suggested dishes based on the ingredients:")
        for idx, dish in enumerate(suggested_dishes):
            st.write(f"{idx + 1}. {dish['name']}")

        for idx, dish in enumerate(suggested_dishes[:3]):
            if st.button(f"Generate Image for Dish {idx + 1}"):
                dish_image = generate_image(dish['name'])
                st.image(dish_image, caption=dish['name'], use_column_width=True)
    else:
        st.write("No dishes found for the given ingredients.")