from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from os import listdir
import time
from keras.models import load_model
from os import environ
import requests
import json
from tensorflow.python.client import device_lib

def process(dirpath="",imgurl=""):
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))


    model = load_model(f"{dirpath}/tl_model_v1.weights.best.h5")

    img_data = requests.get(imgurl).content
    with open('temp.jpg', 'wb') as handler:
        handler.write(img_data)
        handler.truncate()

    classes = []

    with open(f"{dirpath}/meta.json", "r") as f:
        data = json.loads(f.read())
        classes = data["classes"]

    img = image.load_img("temp.jpg", target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    input_preprocessed = preprocess_input(x)
    features_image = model.predict(x, verbose=0)

    feats = features_image[0].flatten()

    with open(f"{dirpath}/last_test.json", "w") as f:
        data = {'img_url':imgurl,
                'predicted_class': classes[np.argmax(feats)]}

        f.write(json.dumps(data,indent=4))
        f.truncate()


