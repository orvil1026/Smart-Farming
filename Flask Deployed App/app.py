import os
from flask import Flask, redirect, render_template, request
from PIL import Image
from keras.models import load_model
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
import cv2
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from tensorflow.keras.utils import img_to_array
import pickle
from rembg import remove
import joblib




disease_info = pd.read_csv('6_disease.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')
pest_info = pd.read_csv('Pest_Detection_1.csv', encoding='cp1252')
weed_info = pd.read_csv('WeedDetection.csv', encoding='cp1252')
# model = CNN.CNN(39)
# model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
# model.eval()


disease_model = load_model(
    'D:\SFIT\SEM 7\Project\Plant-Disease-Detection-main\Plant-Disease-Detection-main\Model\Disease Detection\Vggnet version 5 6 classes\model.h5'
)

pest_model = load_model(
    'D:\SFIT\SEM 7\Project\Plant-Disease-Detection-main\Plant-Disease-Detection-main\Model\Pest Detection\Version2\model.h5'
)

species_model = load_model(
    'D:\SFIT\SEM 7\Project\Plant-Disease-Detection-main\Plant-Disease-Detection-main\Model\Species Recognition\model.h5'
)

weed_model = load_model(
    'D:\SFIT\SEM 7\Project\Plant-Disease-Detection-main\Plant-Disease-Detection-main\Model\Weed Detection\Version2 _ 6 classes resnet\model.h5'
)

crop_prediction_model = joblib.load("D:\SFIT\SEM 7\Project\Plant-Disease-Detection-main\Plant-Disease-Detection-main\Model\Crop Prediction\BayesNaive\model.joblib")

cyp_model = joblib.load("D:\SFIT\SEM 7\Project\Plant-Disease-Detection-main\Plant-Disease-Detection-main\Model\Crop Yield Prediction\cyp.joblib")

# label_binarizer = LabelBinarizer()

# load_binarizer = pickle.open('C:/Users/nashp/Downloads/Plant-Disease-Detection/Plant-Disease-Detection-main/Plant-Disease-Detection-main/Model/label_transform.pkl',)
# n_classes = len(label_binarizer.classes_)
# print(load_binarizer.classes_)

# def prediction(image_path):
#     image = Image.open(image_path)
#     image = image.resize((224, 224))
#     input_data = TF.to_tensor(image)
#     input_data = input_data.view((-1, 3, 224, 224))
#     output = model(input_data)
#     output = output.detach().numpy()
#     index = np.argmax(output)
#     return index

label_binarizer = LabelBinarizer()


def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, tuple((224, 224)))

            #  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None


disease_classes = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
           'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
           'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
           'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
           'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
           'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
           'Tomato_healthy']

disease_classes_6 = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot', 'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus']

pest_classes = ['aphids', 'armyworm', 'beetle', 'bollworm', 'grasshopper', 'mites', 'mosquito', 'sawfly', 'stem_borer']

weed_classes = ['Carpetweeds', 'Eclipta', 'Goosegrass', 'Morningglory', 'Nutsedge','PalmerAmaranth', 'Purslane', 'Sicklepod', 'SpottedSpurge','Waterhemp']

weed_classes_6 = ['Carpetweeds', 'Morningglory', 'Nutsedge', 'PalmerAmaranth', 'Purslane', 'Waterhemp']

species_recognition_classes = ['Apple__healthy', 'Blueberry__healthy',
 'Cherry_(including_sour)__healthy', 'Corn(maize)___healthy',
 'Grape__healthy', 'Peach_healthy', 'Pepper,_bell__healthy',
 'Potato__healthy', 'Raspberry_healthy', 'Soybean__healthy',
 'Strawberry__healthy','Tomato__healthy']

def prediction(image_path, classes_list, model):
    image_array = convert_image_to_array(image_path)
    np_image = np.array(image_array, dtype=np.float32)
    print(np_image.dtype)
    np_image = np.expand_dims(np_image, 0)
    print(np_image.shape)
    # plt.imshow(plt.imread(image_path))

    # result = (model.predict(np_image) > 0.5).astype("int32")

    result = np.argmax(model.predict(np_image), axis=-1)

    # result = model.predict_classes(np_image)
    print(f"result: {int(result)}")
    # print(classes_list[int(result)])
    return int(result)


app = Flask(__name__)


@app.route('/')
def home_page():
    return render_template('home.html')


@app.route('/contact')
def contact():
    return render_template('contact-us.html')


@app.route('/disease-detection')
def disease_detection():
    return render_template('disease-detection.html')

@app.route('/pest-detection')
def pest_detection():
    return render_template('pest-detection.html')

@app.route('/weed-detection')
def weed_detection():
    return render_template('weed-detection.html')

@app.route('/species-recognition')
def species_recognition():
    return render_template('species-recognition.html')

@app.route('/crop-prediction')
def crop_prediction():
    return render_template('crop-prediction.html')

@app.route('/crop-yield-prediction')
def crop_yield_prediction():
    return render_template('crop-yield-prediction.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')


@app.route('/disease-detection-submit', methods=['GET', 'POST'])
def disease_detection_submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)

        print(file_path)

        input_path = file_path
        output_path = f'output.png'

        with open(input_path, 'rb') as i:
            with open(output_path, 'wb') as o:
                input = i.read()
                output = remove(input)
                o.write(output)


        # pred = 1
        # prediction(file_path)
        pred = prediction('output.png', disease_classes_6, disease_model)
        print(f" output: {pred}")
        # title = disease_info['disease_name'][pred]
        title = disease_classes_6[pred]
        print(f"title:{title}")
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html', title=title, desc=description, prevent=prevent,
                               image_url=file_path, pred=pred, sname=supplement_name, simage=supplement_image_url,
                               buy_link=supplement_buy_link)


@app.route('/pest-detection-submit', methods=['GET', 'POST'])
def pest_detection_submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)

        print(file_path)
        # pred = 1
        # prediction(file_path)
        pred = prediction(file_path, pest_classes, pest_model)
        print(f" output: {pred}")
        title = pest_classes[pred]
        print(f"title:{title}")

        description = pest_info['description'][pred]
        prevent = pest_info['Possible Steps'][pred]

        image_url = " "
        supplement_name = " "
        supplement_image_url = " "
        supplement_buy_link =" "
        return render_template('submit.html', title=title, desc=description, prevent=prevent,
                               image_url=file_path, pred=pred, sname=supplement_name, simage=supplement_image_url,
                               buy_link=supplement_buy_link)


@app.route('/weed-detection-submit', methods=['GET', 'POST'])
def weed_detection_submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)

        print(file_path)
        # pred = 1
        # prediction(file_path)
        pred = prediction(file_path, weed_classes_6, weed_model)
        print(f" output: {pred}")
        title = weed_classes_6[pred]
        print(f"title:{title}")
        description = weed_info['description'][pred]
        prevent = weed_info['Possible Steps'][pred]
        image_url = " "
        supplement_name = " "
        supplement_image_url = " "
        supplement_buy_link = " "
        return render_template('submit.html', title=title, desc=description, prevent=prevent,
                               image_url=file_path, pred=pred, sname=supplement_name, simage=supplement_image_url,
                               buy_link=supplement_buy_link)



@app.route('/species-recognition-submit', methods=['GET', 'POST'])
def species_recognition_submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)

        print(file_path)
        # pred = 1
        # prediction(file_path)
        pred = prediction(file_path, species_recognition_classes, species_model)
        print(f" output: {pred}")
        title = species_recognition_classes[pred]
        print(f"title:{title}")
        description = " "
        prevent = " "
        image_url = " "
        supplement_name = " "
        supplement_image_url = " "
        supplement_buy_link = " "
        return render_template('submit.html', title=title, desc=description, prevent=prevent,
                               image_url=file_path, pred=pred, sname=supplement_name, simage=supplement_image_url,
                               buy_link=supplement_buy_link)


@app.route('/crop-prediction-submit', methods=['GET','POST'])
def crop_prediction_submit():
    if request.method == 'POST':
        N = int(request.form.get('inputN'))
        # print(request.form['inputN'])
        P = int(request.form.get('inputP'))
        K = int(request.form.get('inputK'))
        temperature = float(request.form.get('inputTemp'))
        humidity = float(request.form.get('inputHumidity'))
        pH = float(request.form.get('inputPh'))
        rainfall = float(request.form.get('inputRainfall'))

        new_values = [[N, P, K, temperature, humidity, pH, rainfall]]
        X_new = pd.DataFrame(new_values, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

        predictions = crop_prediction_model.predict(X_new)[0]
        predictions_list = predictions.tolist()
        print(f"{N} {P} {K} {temperature} prediction:{type(predictions_list)}")
        print(predictions_list)

        class_index = predictions_list.index(1)
        print(f"{class_index}")

        crop_classes = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
                        'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango',
                        'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas',
                        'pomegranate', 'rice', 'watermelon']

        predicted_class = crop_classes[class_index]





        # print(f"title:{title}")
        description = " "
        prevent = " "
        image_url = " "
        supplement_name = " "
        supplement_image_url = " "
        supplement_buy_link = " "
        return render_template('crop-recommendation-submite.html', title=predicted_class, desc=description, prevent=prevent,
                               image_url=image_url, pred=0, sname=supplement_name, simage=supplement_image_url,
                               buy_link=supplement_buy_link)



@app.route('/crop-yield-prediction-submit', methods=['GET','POST'])
def crop_yield_prediction_submit():
    if request.method == 'POST':
        N = int(request.form.get('inputN'))
        # print(request.form['inputN'])
        P = int(request.form.get('inputP'))
        K = int(request.form.get('inputK'))
        temperature = float(request.form.get('inputTemp'))
        humidity = float(request.form.get('inputHumidity'))
        pressure = float(request.form.get('inputPressure'))
        area = float(request.form.get('inputArea'))
        windspeed = float(request.form.get('inputWindSpeed'))
        district = request.form['district-select']
        season = request.form['season-select']
        crops = request.form['cropSelect']
        soil = request.form['soil-select']

        data = {'area': 0, 'temperature': 0, 'wind_speed': 0, 'pressure': 0, 'humidity': 0, 'N': 0,
                'P': 0, 'K': 0, 'AHMEDNAGAR': 0, 'AKOLA': 0, 'AMRAVATI': 0, 'AURANGABAD': 0, 'BEED': 0, 'BHANDARA': 0,
                'BULDHANA': 0, 'CHANDRAPUR': 0, 'DHULE': 0, 'GADCHIROLI': 0, 'GONDIA': 0, 'HINGOLI': 0, 'JALGAON': 0,
                'JALNA': 0, 'KOLHAPUR': 0, 'LATUR': 0, 'NAGPUR': 0, 'NANDED': 0, 'NANDURBAR': 0, 'NASHIK': 0,
                'OSMANABAD': 0, 'PALGHAR': 0, 'PARBHANI': 0, 'PUNE': 0, 'RAIGAD': 0, 'RATNAGIRI': 0, 'SANGLI': 0,
                'SATARA': 0, 'SINDHUDURG': 0, 'SOLAPUR': 0, 'THANE': 0, 'WARDHA': 0, 'WASHIM': 0, 'YAVATMAL': 0,
                'Kharif': 0, 'Rabi': 0, 'Summer': 0, 'Whole Year': 0, 'Arhar/Tur': 0, 'Bajra': 0, 'Castor seed': 0,
                'Cotton(lint)': 0, 'Gram': 0, 'Groundnut': 0, 'Jowar': 0, 'Linseed': 0, 'Maize': 0,
                'Moong(Green Gram)': 0, 'Niger seed': 0, 'Other Rabi pulses': 0, 'Other Cereals & Millets': 0,
                'Other Kharif pulses': 0, 'Ragi': 0, 'Rapeseed &Mustard': 0, 'Rice': 0, 'Safflower': 0, 'Sesamum': 0,
                'Soyabean': 0, 'Sugarcane': 0, 'Sunflower': 0, 'Tobacco': 0, 'Urad': 0, 'Wheat': 0, 'other oilseeds': 0,

                'Maharashtra': 1, 'chalky':0, 'clay':0,'loamy':0, 'peaty':0, 'sandy':0, 'silt':0, 'silty': 0}

        scaled_area = (area - 1.0) / (711300.0 - 1.0)


        data['area'] = scaled_area
        data['temperature'] = temperature
        data['wind_speed'] = windspeed
        data['N'] = N
        data['P'] = P
        data['K'] = K
        data['humidity'] = humidity
        data['pressure'] = pressure
        data['wind_speed'] = windspeed
        data[district] = 1
        data[season] = 1
        data[crops] = 1
        data[soil] = 1


        feature_list_cyp = [list(data.values())]
        print(feature_list_cyp)
        print(len(feature_list_cyp))
        print(data)
        crop_yield = cyp_model.predict(feature_list_cyp)[0]
        print(crop_yield)
        print(f"district:{district}, season:{season}, crops:{crops}, soil:{soil}, pressure:{pressure}, humidity:{humidity} windspeed:{windspeed}, area:{area}")

        production = area*crop_yield
        return render_template('cyp-submit.html', title="Crop Yield Prediction",cyield=crop_yield, production=production)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']),
                           disease=list(disease_info['disease_name']), buy=list(supplement_info['buy link']))


if __name__ == '':
    app.debug = True
    app.run()
