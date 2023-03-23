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
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

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
st.write("Upload an image of food ingredients in your fridge and get recipe suggestions!")

# Upload the image and extract ingredients (use the appropriate function)
uploaded_image = st.file_uploader("Upload an image of your fridge", type=['jpg', 'jpeg'])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    ingredients = extract_ingredients(image)

    # Generate dish suggestions
    suggested_dishes = generate_dishes(ingredients)

    for i, dish in enumerate(suggested_dishes):
        st.write(f"Suggested Dish {i + 1}: {dish}")

        if st.button(f"Generate Image for Dish {i + 1}"):
            dish_image = generate_image(dish)
            st.image(dish_image, caption=f'Generated Image for {dish}.', use_column_width=True)

            download_link = get_image_download_link(dish_image, f"{dish}.jpg", f"Download {dish} Image")
            st.markdown(download_link, unsafe_allow_html=True)