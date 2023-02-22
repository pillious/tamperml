
# Typing
# Import Libraries 

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import PIL
import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib
import pickle
from PIL import Image, ImageChops, ImageEnhance
from skimage.io import imread
from skimage import exposure, color
from skimage.transform import resize
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from itertools import chain
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from numpy import save,load


from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.optimizers import Adam,SGD
from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import array_to_img, img_to_array
from tensorflow.keras.preprocessing.image import load_img
# tensorflow version 2.9.0
# segmentation_models version
from keras.models import load_model

from keras.applications.resnet import ResNet50
from keras.applications.resnet import ResNet101

from sklearn.metrics import roc_curve, auc,roc_auc_score

from tqdm import tqdm
import cv2

from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers import concatenate

print(tf.__version__)

path_original = './CASIA2.0_revised/Au/'
path_tampered = './CASIA2.0_revised/Tp/'
path_mask = './CASIA2.0_Groundtruth/'
dataset_path = './CASIA2.0_revised/'
# total_original = os.listdir(path_original)
# total_tampered = os.listdir(path_tampered)

images = []

# read in image paths from triplets csv
df = pd.read_csv('ids.csv')

pristine_images = list(df['source'])
fake_images = list(df['tampered'])
mask_images = list(df['target'])

# def mask_pristine(path):
#     img = Image.open(path).convert("RGB")
#     img_shape=(np.array(img)).shape
#     return np.ones((img_shape))*255

# def plot_ground_truth_mask(original_path, mask_path):
#     PATH_ori=path_original + original_path
#     PATH_mask=path_mask + mask_path
    
#     img = Image.open(PATH_ori).convert("RGB")
    
#     try:
#         mask_img=Image.open(PATH_mask).convert("RGB")
#     except:
#         mask_img=mask_pristine(PATH_ori)
#     fig = plt.figure(figsize=(15,10))
#     ax1 = fig.add_subplot(221)
#     ax2 = fig.add_subplot(222)
#     ax1.set_title("Image")
#     ax2.set_title("Ground Truth Mask")
#     ax1.imshow(img)
#     ax2.imshow(mask_img)

# plot_ground_truth_mask(pristine_images[0], mask_images[0])

# Code for Resizing images
# if not os.path.exists(dataset_path+"resized_images/"):
#     os.makedirs(dataset_path+"resized_images/fake_masks/")
#     os.makedirs(dataset_path+"resized_images/image/fake_images/")
#     os.makedirs(dataset_path+"resized_images/image/pristine_images/")
#     height = 256
#     width = 384
#     for fake_image in fake_images:   
#         img=Image.open(path_tampered + fake_image).convert("RGB")
        
#         img = img.resize((width, height), PIL.Image.ANTIALIAS)
#         img.save(dataset_path+"resized_images/image/fake_images/"+fake_image)

#     for mask_image in mask_images:   
#         img=Image.open(path_mask + mask_image).convert("RGB")
        
#         img = img.resize((width, height), PIL.Image.ANTIALIAS)
#         img.save(dataset_path+"resized_images/fake_masks/"+mask_image)

#     for pristine_image in pristine_images:
#         img=Image.open(path_original + pristine_image).convert("RGB")
        
#         img = img.resize((width, height), PIL.Image.ANTIALIAS)
#         img.save(dataset_path+"resized_images/image/pristine_images/"+pristine_image)
# else:
#     print('images resized,path exists')

#https://gist.github.com/cirocosta/33c758ad77e6e6531392
#error level analysis of an image
# def ELA(img_path):
#     """Performs Error Level Analysis over a directory of images"""
    
#     TEMP = 'ela_' + 'temp.jpg'
#     SCALE = 10
#     original = Image.open(img_path)
#     try:
#         original.save(TEMP, quality=90)
#         temporary = Image.open(TEMP)
#         diff = ImageChops.difference(original, temporary)
        
#     except:
        
#         original.convert('RGB').save(TEMP, quality=90)
#         temporary = Image.open(TEMP)
#         diff = ImageChops.difference(original.convert('RGB'), temporary)
        
       
#     d = diff.load()
    
#     WIDTH, HEIGHT = diff.size
#     for x in range(WIDTH):
#         for y in range(HEIGHT):
#             d[x, y] = tuple(k * SCALE for k in d[x, y])
# #     save_path = dataset_path +'ELA_IMAGES/'
# #     diff.save(save_path+'diff.png')
#     return diff

# resized_fake_path = dataset_path+"resized_images/image/fake_images/"

# if not os.path.exists(dataset_path+'ELA_IMAGES/'):
#     os.makedirs(dataset_path+'ELA_IMAGES/')
#     for i in tqdm(fake_images):
#         ELA(resized_fake_path+i).save(dataset_path+'ELA_IMAGES/'+i)
# else:
#     print('Images are already converted to ELA')



ELA_images_with_path = [dataset_path+'ELA_IMAGES/'+i for i in os.listdir(dataset_path+'ELA_IMAGES/') ]
fake_mask_with_path = [dataset_path+"resized_images/fake_masks/"+i for i in os.listdir(dataset_path+"resized_images/fake_masks/") ]

X_train, X_val, Y_train, Y_val = train_test_split(ELA_images_with_path,fake_mask_with_path , test_size=0.12, random_state=7)
X_train = X_train[:1000]
Y_train = Y_train[:1000]
X_val = X_val[:75]
Y_val = Y_val[:75]

# with open("X_train.txt", "wb") as f:   #Pickling
#     pickle.dump(X_train, f) 

# ## save all the converted text into a text file using pickle
# with open("Y_train.txt", "wb") as f:   #Pickling
#     pickle.dump(Y_train, f) 
    
# with open("X_val.txt", "wb") as f:   #Pickling
#     pickle.dump(X_val, f) 

# ## save all the converted text into a text file using pickle
# with open("Y_val.txt", "wb") as f:   #Pickling
#     pickle.dump(Y_val, f) 


def metric(y_true, y_pred, smooth=1): # Dice_Coeff or F-Score
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def LoadImages(batch):
    return np.array([resize(imread(file_name), (384, 256, 3)) for file_name in batch])

#https://stackoverflow.com/questions/47200146/keras-load-images-batch-wise-for-large-dataset
def loadImagesBatchwise(X_train,Y_train, batch_size):
    train_image_files=X_train
    train_mask_files=Y_train
    L = len(train_image_files)
    while True:
        batch_start = 0
        batch_end = batch_size
        
        while batch_start < L:
            limit = min(batch_end, L)
            X = LoadImages(train_image_files[batch_start:limit])
            Y = LoadImages(train_mask_files[batch_start:limit])
            yield (X,Y)
            batch_start += batch_size
            batch_end += batch_size

import segmentation_models as sm
sm.set_framework('keras')
model = sm.Unet('resnet101', input_shape=(384, 256, 3), classes=3, activation='sigmoid',encoder_weights='imagenet')

model.compile(optimizer=optimizers.Adam(), loss="binary_crossentropy", metrics=[metric])
model.summary()

print(len(X_train), len(Y_train), len(X_val), len(Y_val))

batch_size=4
num_training_samples=len(X_train)
num_validation_samples=len(X_val)
# num_epochs=20
num_epochs=5
os.makedirs('model_checkpoints')
# define callbacks for learning rate scheduling and best checkpoints saving
filepath = 'model_checkpoints/model_phase_2.hdf5'
checkpoint = keras.callbacks.ModelCheckpoint(filepath,monitor='val_metric',save_best_only=True, mode='max')

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.22, patience = 1, verbose = 1, min_delta = 0.0001)

# train model
results=model.fit_generator(loadImagesBatchwise(X_train,Y_train,batch_size),steps_per_epoch=(num_training_samples // batch_size), epochs=num_epochs,
                            validation_data=loadImagesBatchwise(X_val, Y_val,batch_size),validation_steps=num_validation_samples//batch_size,
                         verbose=1,callbacks=[early_stop,reduce_lr,checkpoint])

model.save('new_model_phase2.hdf5')







