from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from keras.models import load_model
from PIL import Image, ImageChops
import numpy as np
import base64
import os
import uuid

app = Flask(__name__)
CORS(app)

model = None

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
    if isinstance(d[0, 0], int):
        # if image has single channel, convert to RGB
        diff = diff.convert('RGB')
        d = diff.load()

    WIDTH, HEIGHT = diff.size
    for x in range(WIDTH):
        for y in range(HEIGHT):
            d[x, y] = tuple(k * SCALE for k in d[x, y])

    return diff


def generate_guids(num) -> 'list[str]':
    guids: 'list[str]' = []
    for _ in range(num):
        guids.append(str(uuid.uuid4()))
    return guids


def save_images(base64strs: 'list[str]') -> 'list[str]':
    guids = generate_guids(len(base64strs))

    for i, s in enumerate(base64strs):
        with open(f'images/{guids[i]}.png', "w+b") as fh:
            fh.write(base64.b64decode(s))

    return guids


def delete_images(ids: 'list[str]'):
    for id in ids:
        os.remove(f'images/{id}.png')


def predict(ids: 'list[str]'):
    global model

    paths = list(map(lambda id: f'images/{id}.png', ids))

    x_casia = [np.array(ELA(p).resize((128, 128))
                        ).flatten() / 255.0 for p in paths]
    x_casia = np.array(x_casia)
    x_casia = x_casia.reshape(-1, 128, 128, 3)

    if model == None:
        model = load_model('assets/new_model_casia.h5')
    
    pred = model.predict(x_casia)
    pred_classification = np.argmax(pred, axis=1)
    pred_confidence = np.max(pred, axis=1)

    resp = {'data': []}
    for i, j in zip(pred_classification, pred_confidence):
        resp['data'].append({'isTampered': True if int(
            i) == 1 else False, 'confidence': float(j)})

    return resp


@app.route("/")
def index():
    return "API is running"


@app.route('/analyze', methods=['POST'])
def analyze():
    args = request.get_json()
    base64strs: 'list[str]' = args.get('files')

    if (len(base64strs) == 0):
        return jsonify({'data': []})

    ids = save_images(base64strs)
    resp = predict(ids)
    delete_images(ids)

    return jsonify(resp)
