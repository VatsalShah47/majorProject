from __future__ import print_function
import json
from flask_cors import CORS
from fpdf import FPDF
from flask import Flask, request, render_template, Markup , jsonify
import numpy as np
import pickle
import pandas as pd
from disease import disease_dic
from fertilizer import fertilizer_dic
import requests
import io
import torch
from torchvision import transforms
from PIL import Image
from model import ResNet9
from crop_predict import Crop_Predict
import os
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import openai
import datetime

import pymongo

from dotenv import load_dotenv
import os

load_dotenv()


# fruit disease prediction

import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

openai.api_key = "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" #os.getenv("OPENAI_API_KEY_IMAGE")

client = pymongo.MongoClient("mongodb+srv://pyflask:135792468@farming-assisstant.z38uqhw.mongodb.net/?retryWrites=true&w=majority")
test_db = client["test"]

app = Flask(__name__)
CORS(app)

# Model saved with Keras model.save()
MODEL_PATH ='./test.h5'

# Load your trained model
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(512, 512))
    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)


    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="Brownspot"
    elif preds==1:
        preds="Healthy"
    else :
        preds="Woodiness"


    return preds


@app.route('/predict-fruit-disease', methods=["POST"])
def upload():
    if request.method == "POST":
        # Get the file from post request
        # print("dfdsf")
        # print(request.files)
        # if "file" not in request.files:
        #     return "file not found"
        file = request.files.get("file")

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(file.filename))
        file.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        print(result)
        return result

        # # Save the file to ./uploads

        # # f.save(file_path)

        # # Make prediction

    return "hello"


# Loading plant disease classification model
disease_classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


disease_model_path = "models/plant-disease-model.pth"
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(
    torch.load(disease_model_path, map_location=torch.device("cpu"))
)
disease_model.eval()

disease_info = pd.read_csv("disease_info.csv", encoding="cp1252")
supplement_info = pd.read_csv("supplement_info.csv", encoding="cp1252")

model = CNN.CNN(39)
model.load_state_dict(torch.load("models/diseaseV2.pt"))
model.eval()


def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index


# prediction function
def CropPredictor(to_predict_list):
    to_predict = np.array([to_predict_list])
    loaded_model = pickle.load(open("models/RandomForest.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]


def FertilizerPredictor(to_predict_list):
    to_predict = np.array([to_predict_list])
    loaded_model = pickle.load(open("models/classifier.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]


def WeatherPredictor(to_predict_list):
    to_predict = np.array([to_predict_list])
    loaded_model = pickle.load(open("models/weather.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]


def DiseasesPredictor(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    print(prediction)
    return prediction


# routing
@app.route("/", methods=["GET"])
def home():
    return "server started..."


# @app.route("/crop-predict2", methods=["POST"])
# def result():
#     if request.method == "POST":
#         print(request.json)
#         to_predict_list = request.json
#         to_predict_list = list(to_predict_list.values())
#         to_predict_list = list(map(int, to_predict_list))
#         result = CropPredictor(to_predict_list)
#         return result


@app.route("/crop-predict", methods=["POST"])
def crop():
    model = Crop_Predict()
    if request.method == "POST":
        crop_name = model.crop()
        if crop_name == "noData":
            return -1

        return {
            "crop_name": crop_name,
            "no_of_crops": len(crop_name),
        }


@app.route("/fertilizer-predict", methods=["POST"])
def result2():
    if request.method == "POST":
        print(request.json)
        to_predict = request.json
        location = request.json["location"]
        del to_predict["location"]

        to_predict_list = list(to_predict.values())

        # Use the OpenWeatherMap API to get the weather forecast for the next 15 days
        api_key = "910fc3efa3d910c22b2a8a6a6989347c" #os.getenv("OPEN_WEATHER_API_KEY")
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={location}&appid={api_key}"
        response = requests.get(url)
        weather_data = response.json()

        print((float(weather_data["list"][0]["main"]["temp"]) - 273.15))
        Temp = float(weather_data["list"][0]["main"]["temp"]) - 273.15
        Hum = weather_data["list"][0]["main"]["humidity"]
        to_predict_list.append(Temp)
        to_predict_list.append(Hum)
        print(to_predict_list)

        to_predict_list = list(map(int, to_predict_list))

        ans = FertilizerPredictor(to_predict_list)

        fertilizer_info = {"name": "", "img": ""}
        if ans == 0:
            test_db["fertilizer_recommendation"].insert_one({"input":to_predict,"output":"Compost"})
            response = openai.Image.create(
                prompt="compost from food scraps, yard waste",
                n=1,
                size="256x256",
            )
            return {
                "name": "Compost",
                "img": response["data"][0]["url"],
                "how_to_use": "Compost is easy to make at home using food scraps, yard waste, and other organic materials. You can also purchase compost at garden centers and nurseries. To use compost as a fertilizer, simply mix it into the soil before planting or use it as a top dressing around established plants. \nThat being said, it's always a good idea to do a soil test to determine the specific nutrient needs of your plants and soil. This can help you choose the right organic fertilizer and ensure that your plants are getting the nutrients they need to grow and thrive.",
            }
        elif ans == 1:
            test_db["fertilizer_recommendation"].insert_one({"input":to_predict,"output":"Dr. Earth Organic 5 Tomato, Vegetable & Herb Fertilizer"})

            response = openai.Image.create(
                prompt="Dr. Earth Organic 5 Tomato, Vegetable & Herb Fertilizer",
                n=1,
                size="256x256",
            )
            return {
                "name": "Dr. Earth Organic 5 Tomato, Vegetable & Herb Fertilizer",
                "img": response["data"][0]["url"],
                "how_to_use": "Dr. Earth Organic 5 Tomato, Vegetable & Herb Fertilizer organic components: Fish bone meal | Alfalfa meal | Feather meal | Soft rock phosphate | Mined potassium sulfate | Seaweed extract | Beneficial soil microbes",
            }
        elif ans == 2:
            test_db["fertilizer_recommendation"].insert_one({"input":to_predict,"output":"Dr. Earth All Purpose Fertilizer"})

            response = openai.Image.create(
                prompt="Dr. Earth All Purpose Fertilizer",
                n=1,
                size="256x256",
            )
            return {
                "name": "Dr. Earth All Purpose Fertilizer",
                "img": response["data"][0]["url"],
                "how_to_use": "Dr. Earth All Purpose Fertilizer organic components: Feather meal | Seaweed extract | Soft rock phosphate | Humic acid | Kelp meal | Blood meal | Bone meal | Dolomite lime",
            }
        elif ans == 3:
            test_db["fertilizer_recommendation"].insert_one({"input":to_predict,"output":"Jobe's Organics All-Purpose Fertilizer"})

            response = openai.Image.create(
                prompt="Jobe's Organics All-Purpose Fertilizer",
                n=1,
                size="256x256",
            )
            return {
                "name": "Jobe's Organics All-Purpose Fertilizer",
                "img": response["data"][0]["url"],
                "how_to_use": "Jobe's Organics All-Purpose Fertilizer organic composition: Feather meal | Bone meal | Sulfate of potash | Kelp meal | Alfalfa meal | Humic acid",
            }
        elif ans == 4:
            test_db["fertilizer_recommendation"].insert_one({"input":to_predict,"output":"Dr. Earth Organic Nitrogen Fertilizer"})

            response = openai.Image.create(
                prompt="Dr. Earth Organic Nitrogen Fertilizer",
                n=1,
                size="256x256",
            )
            return {
                "name": "Dr. Earth Organic Nitrogen Fertilizer",
                "img": response["data"][0]["url"],
                "how_to_use": "Dr. Earth Organic Nitrogen Fertilizer organic composition: Soybean meal | Alfalfa meal | Fishbone meal | Feather meal | Seabird guano | Blood meal | Kelp meal | Potassium sulfate | Humic acid",
            }
        elif ans == 5:
            test_db["fertilizer_recommendation"].insert_one({"input":to_predict,"output":"Espoma Organic Lawn Food"})

            response = openai.Image.create(
                prompt="Espoma Organic Lawn Food",
                n=1,
                size="256x256",
            )
            return {
                "name": "Espoma Organic Lawn Food",
                "img": response["data"][0]["url"],
                "how_to_use": "Espoma Organic Lawn Food organic composition: Corn gluten meal | Feather meal | Soybean meal | Potassium sulfate | Humates | Iron",
            }
        else:
            test_db["fertilizer_recommendation"].insert_one({"input":to_predict,"output":"FoxFarm"})

            response = openai.Image.create(
                prompt="FoxFarm",
                n=1,
                size="256x256",
            )
            return {
                "name": "FoxFarm",
                "img": response["data"][0]["url"],
                "how_to_use": "FoxFarm organic composition: Earthworm castings | Bat guano | Fish meal | Bone meal | Blood meal | Feather meal | Kelp meal",
            }

@app.route("/fertilizer-predict", methods=["GET"])
def get_fertilizer_predict():
    # data = test_db["fertilizer_recommendation"].find()
    fertilizers = ["Compost","Dr. Earth Organic 5 Tomato, Vegetable & Herb Fertilizer","Dr. Earth All Purpose Fertilizer","Jobe's Organics All-Purpose Fertilizer","Dr. Earth Organic Nitrogen Fertilizer","Espoma Organic Lawn Food","FoxFarm"]
    results = []
    for fertilizer in fertilizers:
        i = test_db["fertilizer_recommendation"].count_documents({"output":fertilizer})
        results.append(i if i is not None else 0)
    result = {
        "labels":fertilizers,
        "datasets":results,
        "xlabel":"Fertilizers",
        "ylabel":"count"
    }
    result = jsonify(result)
    return result


@app.route("/weather-predict", methods=["POST"])
def result3():
    if request.method == "POST":
        # print(request.json)
        to_predict_list = request.json
        # to_predict_list = list(to_predict_list.values())
        to_predict_list = list(to_predict_list.values())
        weather = WeatherPredictor(to_predict_list)
        result = {"data": weather}
        return result


@app.route("/disease-predict", methods=["POST"])
def disease_prediction():
    title = " Disease Detection"

    if request.method == "POST":
        print(request.files)
        if "file" not in request.files:
            return "file not found"
        file = request.files.get("file")
        if not file:
            return "plz send image"
        try:
            img = file.read()
            print(file)

            prediction = DiseasesPredictor(img)
            print(prediction)

            prediction = Markup(str(disease_dic[prediction]))
            return {"prediction": prediction, "title": title}
        except:
            pass
    return render_template("disease.html", title=title)


@app.route("/disease-predict", methods=["GET"])
def get_disease_predict():
    # data = test_db["fertilizer_recommendation"].find()
    results = []
    for disease in range(0,38):
        try:
            i = test_db["fertilizer_recommendation"].count_documents({"disease":disease})
        except:
            pass
        results.append(i if i is not None else 0)
    result = {
        "labels":disease_classes,
        "datasets":results,
        "xlabel":"Fertilizers",
        "ylabel":"count"
    }
    result = jsonify(result)
    return result

@app.route("/disease-predict2", methods=[ "POST"])
def submit():
    if request.method == "POST":
        image = request.files["file"]
        filename = image.filename
        # file_path = os.path.join("static/uploads", filename)
        # image.save(file_path)
        # print(file_path)
        pred = prediction(image)
        title = disease_info["disease_name"][pred]
        description = disease_info["description"][pred]
        prevent = disease_info["Possible Steps"][pred]
        image_url = disease_info["image_url"][pred]
        supplement_name = supplement_info["supplement name"][pred]
        supplement_image_url = supplement_info["supplement image"][pred]
        supplement_buy_link = supplement_info["buy link"][pred]
        print(pred)
        # openai.api_key = os.getenv("OPENAI_API_KEY")
        instructions = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"how to use {supplement_name}",
            max_tokens=200,
            temperature=0,
        )
        print(instructions)
        crop , disease = title.split(":")
        test_db["disease_predict"].insert_one({"disease": pred})

        return {
            "title": title,
            "desc": description,
            "prevent": prevent,
            # "image_url": image_url,
            # "pred": pred,
            "sname": supplement_name,
            "simage": supplement_image_url,
            "buy_link": supplement_buy_link,
            "how_to_use": instructions.choices[0].text,
        }


@app.route("/price-predict", methods=["POST"])
def result4():
    if request.method == "POST":
        print(request.json)
        to_predict_list = request.json
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = CropPredictor(to_predict_list)
        return result


@app.route("/forecast", methods=["POST"])
def forecast():
    # Get the user's location from the form
    location = request.json["location"]

    # Use the OpenWeatherMap API to get the weather forecast for the next 15 days
    # api_key = os.getenv("OPEN_WEATHER_API_KEY")
    api_key = "25a7391eb816518d0639ab3f83a31f42"
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={location}&cnt=15&appid={api_key}"
    response = requests.get(url)
    weather_data = response.json()

    # Extract the necessary information from the API response
    forecast = []
    for item in weather_data["list"]:
        forecast.append(
            {
                "date": item["dt_txt"],
                "temperature": item["main"]["temp"],
                "humidity": item["main"]["humidity"],
                "wind": item["wind"]["speed"],
            }
        )

    month = datetime.datetime.now().month
    hemisphere = "north"

    # Determine the season based on the month and hemisphere
    if (month >= 3 and month <= 6) and hemisphere == "north":
        climate = "summer"
    elif (month >= 7 and month <= 10) and hemisphere == "north":
        climate = "rainy"
    elif (
        month == 11 or month == 12 or month == 1 or month == 2
    ) and hemisphere == "north":
        climate = "winter"

    temperature = forecast[0]["temperature"]
    openai.api_key = os.getenv("OPENAI_API_KEY")
    instructions = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"aggricultural conditions based on {temperature} kelvin and {climate} climate",
        max_tokens=1000,
        temperature=0,
    )
    analysis = instructions.choices[0].text
    forecast = json.dumps(forecast)
    # Return the forecast to the user
    return [forecast, analysis]


@app.route("/getnews", methods=["GET"])
def getnews():
    api_key = "5e1392e4a78241adbf27393420e62ec2"
    base_url = "https://newsapi.org/v2/everything?"
    query = "agriculture+India"
    sources = "bbc-news,the-hindu,the-times-of-india,ndtv"
    language = "en"
    sortBy = "relevancy"
    pageSize = 100

    complete_url = f"{base_url}q={query}&sources={sources}&language={language}&sortBy={sortBy}&pageSize={pageSize}&apiKey={api_key}"

    response = requests.get(complete_url)
    news_data = response.json()
    articles = news_data.get("articles")

    return articles


if __name__ == "__main__":
    app.run()
