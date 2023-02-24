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


# @app.route('/', methods=['GET'])
@app.route("/")
def index():
    # args = request.args
    # playlistIds = args.get("playlistIds").split(",")

    x_casia = []
    x_casia.append(
        np.array(ELA('test.png').resize((128, 128))).flatten() / 255.0)
    x_casia = np.array(x_casia)
    x_casia = x_casia.reshape(-1, 128, 128, 3)

    model = load_model('new_model_casia.h5')
    pred = model.predict(x_casia)
    pred_classification = np.argmax(pred, axis=1)
    pred_confidence = np.max(pred, axis=1)

    resp = {'data': []}
    for i, j in zip(pred_classification, pred_confidence):
        resp['data'].append({'isTampered': True if int(
            i) == 1 else False, 'confidence': float(j)})

    return jsonify(resp)


@app.route('/upload', methods=['POST'])
def upload():
    args = request.get_json()
    print("TEST")
    print(args)
    print(args.get('file'))
    return jsonify({'data': 'success'})
