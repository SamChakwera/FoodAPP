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

def extract_ingredients(image):
    # Save the PIL Image as a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        image.save(temp_file, format="JPEG")
        temp_file_path = temp_file.name

    preds = classifier(temp_file_path)
    predictions = [pred["label"] for pred in preds]
    return [prediction for prediction in predictions if prediction in ingredient_names]

def generate_dishes(ingredients, n=3, max_tokens=150, temperature=0.7):
    ingredients_str = ', '.join(ingredients)
    prompt = f"I have {ingredients_str} Please return the name of a dish I can make followed by intructions on how to prepare that dish "

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n
    )

    dishes = [choice.text.strip() for choice in response.choices]
    return dishes

def generate_image(prompt):
    model_id = "runwayml/stable-diffusion-v1-5"
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

uploaded_image = st.file_uploader("Upload an image of your fridge", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Fridge Image, Please wait", use_column_width=True)

    ingredients = extract_ingredients(image)
    st.write("Detected Ingredients:")
    st.write(ingredients)

    suggested_dishes = generate_dishes(ingredients)

    if suggested_dishes:
        st.write("Suggested Dishes:")
        for dish in suggested_dishes:
            st.write(dish)

        if st.button("Generate Image for Dish 1"):
            dish1_image = generate_image(suggested_dishes[0].split(":")[0])
            st.image(dish1_image, caption=f"Generated Image for {suggested_dishes[0].split(':')[0]}", use_column_width=True)

        if st.button("Generate Image for Dish 2"):
            dish2_image = generate_image(suggested_dishes[1].split(":")[0])
            st.image(dish2_image, caption=f"Generated Image for {suggested_dishes[1].split(':')[0]}", use_column_width=True)

        if st.button("Generate Image for Dish 3"):
            dish3_image = generate_image(suggested_dishes[2].split(":")[0])
            st.image(dish3_image, caption=f"Generated Image for {suggested_dishes[2].split(':')[0]}", use_column_width=True)
    else:
        st.write("No dishes found")
else:
    st.write("Please upload an image")