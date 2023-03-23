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


openai.api_key = os.getenv("OPENAI_API_KEY")

# Load models and set up GPT-3 pipeline
extractor = AutoFeatureExtractor.from_pretrained("stchakman/Fridge_Items_Model")
model = AutoModelForImageClassification.from_pretrained("stchakman/Fridge_Items_Model")
#gpt3 = pipeline("text-davinci-003", api_key="your_openai_api_key")

# Map indices to ingredient names
term_variables = {
    "Apples",
    "Asparagus", 
    "Avocado", 
    "Bananas", 
    "BBQ sauce", 
    "Beans", 
    "Beef", 
    "Beer", 
    "Berries", 
    "Bison", 
    "Bread", 
    "Broccoli", 
    "Cauliflower", 
    "Celery", 
    "Cheese", 
    "Chicken", 
    "Chocolate", 
    "Citrus fruits", 
    "Clams", 
    "Cold cuts", 
    "Corn", 
    "Cottage cheese", 
    "Crab", 
    "Cream", 
    "Cream cheese", 
    "Cucumbers", 
    "Duck", 
    "Eggs", 
    "Energy drinks", 
    "Fish", 
    "Frozen vegetables", 
    "Frozen meals", 
    "Garlic", 
    "Grapes", 
    "Ground beef", 
    "Ground chicken", 
    "Ham", 
    "Hot sauce", 
    "Hummus", 
    "Ice cream", 
    "Jams", 
    "Jerky", 
    "Kiwi", 
    "Lamb", 
    "Lemons", 
    "Lobster", 
    "Mangoes", 
    "Mayonnaise", 
    "Melons", 
    "Milk", 
    "Mussels", "Mustard", "Nectarines", "Onions", "Oranges", "Peaches", "Peas", "Peppers", "Pineapple", "Pizza", "Plums", "Pork", "Potatoes", "Salad dressings", "Salmon", "Shrimp", "Sour cream", "Soy sauce", "Spinach", "Squash", "Steak", "Sweet potatoes", "Frozen Fruits", "Tilapia", "Tomatoes", "Tuna", "Turkey", "Venison", "Water bottles", "Wine", "Yogurt", "Zucchini"
}
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

def generate_images(dishes):
    prompt = f"Generate an image for each of the following dishes: {dishes[0]}, {dishes[1]}, {dishes[2]}."

    response = openai.Image.create_edit(
        image=None,  # You will need to provide the image input
        mask=None,   # You will need to provide the mask input, if required
        prompt=prompt,
        n=3,
        size="256x256",
        response_format="url",
    )

    images = []
    for url in response['data']:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        images.append(image)

    return images

st.title("Fridge to Dish App")
st.write("Upload an image of food ingredients in your fridge and get recipe suggestions!")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    ingredients = extract_ingredients(image)
    st.write(f"Ingredients detected: {', '.join(ingredients)}")

    suggested_dishes = generate_dishes(ingredients)
    st.write("Suggested dishes:")
    st.write(suggested_dishes)

    dish_images = generate_images(suggested_dishes)

    # Display dish images in a grid
    # Replace the following lines with code to display generated images
    st.write("Generated images:")
    for i in range(3):
        st.image("placeholder.jpg", caption=f"Dish {i+1}")