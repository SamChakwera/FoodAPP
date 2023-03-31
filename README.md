# Fridge 2 Dish App

Fridge to Dish App is a web application that helps users find dish suggestions based on the ingredients they have in their fridge. Users can upload an image of their ingredients, and the app will identify the ingredients and provide dish suggestions along with their instructions. Users can also generate an image for each suggested dish.

## Features

- Ingredient recognition from an image
- Dish suggestions based on the ingredients
- Generating images for the suggested dishes

## Installation

1. Clone the repository

```
git clone https://github.com/yourusername/fridge-to-dish-app.git
```

2. Change the working directory

```
cd fridge-to-dish-app
```

3. Create a virtual environment

```
python -m venv venv
```

4. Activate the virtual environment

```
# For Windows
venv\Scripts\activate

# For Linux/Mac
source venv/bin/activate
```

5. Install the dependencies

```
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit app

```
streamlit run app.py
```

2. Open the app in your browser using the provided URL

3. Upload an image of your ingredients

4. Review the detected ingredients and suggested dishes

5. Click on the "Generate Image" button to see an image of each suggested dish

## Technologies Used

- Python
- Streamlit
- OpenAI GPT-3 (API Key required) 
- OpenAI CLIP
- Runwayml Stable Diffusion v1-5
- PIL (Python Imaging Library)
- Diffusers

## License

This project is licensed under the [MIT License](LICENSE).

## Example
You can find a version of this working on my [huggingface space](https://huggingface.co/spaces/stchakman/Fridge2Dish). Feel free to clone it however you will need to run it with atleast the nvidia T4 GPU becaus the diffusion model requires too much resources it cant run on the base CPU. 

## Contact

If you have any questions about this project or would like to contact the maintainer, please send an email to [@samchakwera](https://github.com/SamChakwera).
