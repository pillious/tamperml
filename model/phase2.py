from math import ceil
from segmentation_models import Unet
from albumentations import *
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.optimizers import Adam, SGD
from numpy import save, load
from keras.layers import concatenate
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
import cv2
from keras.applications.resnet import ResNet101
from keras.applications.resnet import ResNet50
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import array_to_img, img_to_array
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras import backend as K
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from itertools import chain
from skimage.morphology import label
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage import exposure, color
from skimage.io import imread
from PIL import Image, ImageChops, ImageEnhance
import pickle
import matplotlib
from PIL import Image
import os
import pandas as pd
import numpy as np
import PIL
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# tensorflow version 2.9.0
# segmentation_models version


path_original = 'dataset-dist/phase-01/training/pristine/'
path_tampered = 'dataset-dist/phase-01/training/fake/'
dataset_path = 'dataset-dist/phase-01/training/'
total_original = os.listdir(path_original)
total_tampered = os.listdir(path_tampered)

total_tampered.remove('.DS_Store')

# saving the path along with the file names
pristine_images = []
for i in total_original:
    pristine_images.append(dataset_path+i)
fake_images = []
for i in total_tampered:
    fake_images.append(dataset_path+i)


def mask_pristine(path):
    img = Image.open(path).convert("RGB")
    img_shape = (np.array(img)).shape
    return np.ones((img_shape))*255


if not os.path.exists(dataset_path+"resized_images/"):
    os.makedirs(dataset_path+"resized_images/fake_masks/")
    os.makedirs(dataset_path+"resized_images/image/fake_images/")
    os.makedirs(dataset_path+"resized_images/image/pristine_images/")
    height = 512
    width = 512
    for fake_image in total_tampered:

        if('.mask' in fake_image):
            img = Image.open(path_tampered + fake_image).convert("RGB")

            img = img.resize((height, width), PIL.Image.ANTIALIAS)
            img.save(dataset_path+"resized_images/fake_masks/"+fake_image)
        else:

            img = Image.open(path_tampered + fake_image).convert("RGB")

            img = img.resize((height, width), PIL.Image.ANTIALIAS)
            img.save(dataset_path+"resized_images/image/fake_images/"+fake_image)

    for pristine_image in total_original:
        img = Image.open(path_original + pristine_image).convert("RGB")

        img = img.resize((height, width), PIL.Image.ANTIALIAS)
        img.save(dataset_path+"resized_images/image/pristine_images/"+pristine_image)


else:
    print('images resized,path exists')

resized_fakes = os.listdir(dataset_path+"resized_images/image/fake_images/")
resized_fake_path = dataset_path+"resized_images/image/fake_images/"

# https://gist.github.com/cirocosta/33c758ad77e6e6531392
# error level analysis of an image


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
#     save_path = dataset_path +'ELA_IMAGES/'
#     diff.save(save_path+'diff.png')
    return diff


if not os.path.exists(dataset_path+'ELA_IMAGES/'):
    os.makedirs(dataset_path+'ELA_IMAGES/')
    for i in resized_fakes:
        ELA(resized_fake_path+i).save(dataset_path+'ELA_IMAGES/'+i)
else:
    print('Images are already converted to ELA')

ELA_images_with_path = [dataset_path+'ELA_IMAGES/' +
                        i for i in os.listdir(dataset_path+'ELA_IMAGES/')]
fake_mask_with_path = [dataset_path+"resized_images/fake_masks/" +
                       i for i in os.listdir(dataset_path+"resized_images/fake_masks/")]
ELA_images_with_path.sort()
fake_mask_with_path.sort()
total_tampered.sort()

X_train, X_val, Y_train, Y_val = train_test_split(
    ELA_images_with_path, fake_mask_with_path, test_size=0.12, random_state=7)

X_tr=X_train[:100]
Y_tr=Y_train[:100]
X_v=X_val[:20]
Y_v=Y_val[:20]

def metric(y_true, y_pred, smooth=1):  # Dice_Coeff or F-Score
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def LoadImages(batch):
    return np.array([resize(imread(file_name), (512, 512, 3)) for file_name in batch])
# https://stackoverflow.com/questions/47200146/keras-load-images-batch-wise-for-large-dataset


# def loadImagesBatchwise(X_train, Y_train, batch_size):
#     train_image_files = X_train
#     train_mask_files = Y_train
#     L = len(train_image_files)
#     while True:
#         batch_start = 0
#         batch_end = batch_size

#         while batch_start < L:
#             limit = min(batch_end, L)
#             X = LoadImages(train_image_files[batch_start:limit])
#             Y = LoadImages(train_mask_files[batch_start:limit])
#             yield (X, Y)
#             batch_start += batch_size
#             batch_end += batch_size


# model = Unet('resnet101', input_shape=(512, 512, 3), classes=3,
#              activation='sigmoid', encoder_weights='imagenet')
# model.compile(optimizer=optimizers.Adam(),
#               loss="binary_crossentropy", metrics=[metric])


# batch_size = 4

# num_training_samples = len(X_tr)
# num_validation_samples = len(X_v)

# # steps = ceil(len(X_train)//batch_size)
# num_epochs = 5
# os.makedirs('model_checkpoints')
# # define callbacks for learning rate scheduling and best checkpoints saving
# filepath = 'model_checkpoints/model_phase_2.hdf5'
# checkpoint = keras.callbacks.ModelCheckpoint(
#     filepath, monitor='val_metric', save_best_only=True, mode='max')

# early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# reduce_lr = ReduceLROnPlateau(
#     monitor='val_loss', factor=0.22, patience=1, verbose=1, min_delta=0.0001)

# results = model.fit_generator(loadImagesBatchwise(X_tr, Y_tr, batch_size), steps_per_epoch=(num_training_samples // batch_size), epochs=num_epochs,
#                               validation_data=loadImagesBatchwise(X_v, Y_v, batch_size), validation_steps=num_validation_samples//batch_size,
#                               verbose=1, callbacks=[early_stop, reduce_lr, checkpoint])

# model.save('new_model_phase2.hdf5')

model = load_model('model_checkpoints/model_phase_2.hdf5', {"metric": metric})
print("model loaded")

test_images=LoadImages(X_v)
predicted=model.predict(test_images)

def plot_predicted_images(index):
    """Plots the predicted masks of tampered images"""
    #ret, bw_img = cv2.threshold((predicted[index]*255),127,255,cv2.THRESH_BINARY)
    plt.imsave(f'output/pred_mask{str(i)}.png',predicted[index])
    im_gray = cv2.imread(f'output/pred_mask{str(i)}.png', cv2.IMREAD_GRAYSCALE)
    (thresh, im_bw) = cv2.threshold(im_gray, 220, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #imshow(im_bw)
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(441)
    ax2 = fig.add_subplot(442)
    ax3 = fig.add_subplot(443)
    ax4 = fig.add_subplot(444)
    
    ax1.set_title("actual_image")
    ax2.set_title("actual_mask")
    ax3.set_title("predicted_mask")
    ax4.set_title("binary_predicted_mask")
    actual_img = imread(path_tampered+X_val[index].split('/')[-1])

    temp_path = X_val[index].split('/')[-1]
    print(path_tampered+temp_path)

    actual_mask = imread(Y_val[index])
    # predicted_mask = imread(predicted[0])

    
    ax1.imshow(actual_img)
    ax2.imshow(actual_mask)
    ax3.imshow(predicted[index])
    ax4.imshow(im_bw)

    plt.imsave(f'output/actual_img{str(i)}.png',actual_img)
    plt.imsave(f'output/actual_mask{str(i)}.png',actual_mask)
    plt.imsave(f'output/im_bw{str(i)}.png',im_bw)

for i in range(len(test_images)):
    plot_predicted_images(i)

    



    



