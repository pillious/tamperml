from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from keras.models import Model, load_model
from PIL import Image, ImageChops
import numpy as np

app = Flask(__name__)
CORS(app)


def ELA(img_path):
    """Performs Error Level Analysis over a directory of images"""

    TEMP = 'ela_' + 'temp.jpg'
    SCALE = 10
    original = Image.open(img_path)
    try:
        original.save(TEMP, quality=90)
        temporary = Image.open(TEMP)
        diff = ImageChops.difference(original, temporary)

    except:

        original.convert('RGB').save(TEMP, quality=90)
        temporary = Image.open(TEMP)
        diff = ImageChops.difference(original.convert('RGB'), temporary)

    d = diff.load()

    WIDTH, HEIGHT = diff.size
    for x in range(WIDTH):
        for y in range(HEIGHT):
            d[x, y] = tuple(k * SCALE for k in d[x, y])

    return diff


@app.route("/")
def index():
    x_casia = []
    x_casia.append(np.array(ELA('test.png').resize((128, 128))).flatten() / 255.0)
    x_casia = np.array(x_casia)
    x_casia = x_casia.reshape(-1, 128, 128, 3)

    model = load_model('new_model_casia.h5')
    predictions = model.predict(x_casia)
    predictions = np.argmax(predictions, axis=1)

    # class label 1 ===>tampered
    # class label 0====>real

    print(predictions)

    return "use the /recommend route"
