# -*- coding: utf-8 -*-


###Transfer learning codes for Baseline Models were based on https://www.tensorflow.org/tutorials/images/transfer_learning

import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score,recall_score, f1_score, precision_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,InputLayer,Conv2DTranspose,UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.initializers import HeNormal, GlorotNormal

tf.__version__ #implemented 2.19.0

import kagglehub

kagglehub.__version__ #0.3.13

os.environ['KAGGLEHUB_CACHE'] = '/tmp/kagglehub_cache'

import kagglehub

# Download latest version
path = kagglehub.dataset_download("orvile/acrima-glaucoma-assessment-using-fundus-images")

print("Path to dataset files:", path)

tr_dir = '/tmp/kagglehub_cache/datasets/orvile/acrima-glaucoma-assessment-using-fundus-images/versions/1/Database/Images'

#If already downloaded, use the below code for directory specification
#tr_dir = '/kaggle/input/acrima-glaucoma-assessment-using-fundus-images/Database/Images'

Filename = pd.DataFrame(os.listdir(tr_dir),columns=["filename"])
Filename

Filename = Filename.sort_values(by=['filename'], ascending=True)

Filename.index = [i for i in range(0,705,1)]

Filename

def load_images_and_labels(image,label):
    IMG = []
    LB = []
    for index, row in label.iterrows():
        img_path = os.path.join(image, f"{row['filename']}")
        if os.path.exists(img_path):
            img = cv2.imread(img_path,cv2.IMREAD_COLOR_RGB)
            img = cv2.resize(img, (224, 224)) # Resize images
            IMG.append(img)
            if "_g_" in row['filename']:
              LB.append(1)
            else:
              LB.append(0)
            if len(LB)%100 == 0: print(index+1)
    return np.array(IMG), np.array(LB)

# Loading images and labels
X_1,tag1 = load_images_and_labels(tr_dir,Filename)

np.bincount(tag1)

#Example of dataset
plt.imshow(X_1[1,:,:,:])
print(tag1[1])

###Train/test data

X_1 = X_1/255.0 #rescaling

#reshuffling data
(X_tr,X_te3,tag_tr,tag_te) = train_test_split(X_1,tag1,test_size=0.5,shuffle=True,random_state=321)
print(np.bincount(np.int32(tag_tr)))

plt.imshow(X_tr[5,:,:,:])
print(tag_tr[5])



#################### MobileNetV2

IMG_SIZE = (224,224)
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')

base_model.trainable = False

tf.keras.utils.set_random_seed(987)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
dense1_layer = tf.keras.layers.Dense(64,activation="elu")
prediction_layer = tf.keras.layers.Dense(1, activation="sigmoid")

inputs = tf.keras.Input(shape=(224, 224, 3))

x = base_model(inputs, training=False)
x = global_average_layer(x)
x = dense1_layer(x)
outputs = prediction_layer(x)
model1 = tf.keras.Model(inputs, outputs)

from tensorflow.keras.optimizers import RMSprop, Adam
model1.compile(optimizer=Adam(learning_rate=1e-4), loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'])

model1.fit(X_tr,tag_tr,epochs=300,batch_size=100,shuffle=True)

f1_score(tag_tr,np.round(model1.predict(X_tr)))

accuracy_score(tag_tr,np.round(model1.predict(X_tr)))

prd = model1.predict(X_te3)
prd1 = []
for i in range(len(tag_te)):
  if prd[i] < 0.5:
    prd1.append(0)
  else:
    prd1.append(1)

from sklearn.metrics import ConfusionMatrixDisplay
dis = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(tag_te,prd1), display_labels=[0,1])
dis.plot()
#plt.grid()
plt.show()

from sklearn.metrics import classification_report
print(classification_report(tag_te,prd1,digits=4))

from sklearn.metrics import precision_score, recall_score,f1_score
print(precision_score(tag_te,prd1))
print(recall_score(tag_te,prd1))
print(f1_score(tag_te,prd1))

from sklearn.metrics import roc_auc_score
roc_auc_score(tag_te,model1.predict(X_te3))

model1.summary()

####################### InceptionV3

IMG_SIZE = (224,224)
IMG_SHAPE = IMG_SIZE + (3,)
base_model2 = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')

base_model2.trainable = False

tf.keras.utils.set_random_seed(987)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
dense1_layer = tf.keras.layers.Dense(64,activation="elu")
prediction_layer = tf.keras.layers.Dense(1, activation="sigmoid")

inputs = tf.keras.Input(shape=(224, 224, 3))

x = base_model2(inputs, training=False)
x = global_average_layer(x)
x = dense1_layer(x)
outputs = prediction_layer(x)
model2 = tf.keras.Model(inputs, outputs)

from tensorflow.keras.optimizers import RMSprop, Adam
model2.compile(optimizer=Adam(learning_rate=1e-4), loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'])

model2.fit(X_tr,tag_tr,epochs=300,batch_size=100,shuffle=True)

prd = model2.predict(X_te3)
prd2 = []
for i in range(len(tag_te)):
  if prd[i] < 0.5:
    prd2.append(0)
  else:
    prd2.append(1)

from sklearn.metrics import ConfusionMatrixDisplay
dis = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(tag_te,prd2), display_labels=[0,1])
dis.plot()
#plt.grid()
plt.show()

from sklearn.metrics import classification_report
print(classification_report(tag_te,prd2,digits=4))

from sklearn.metrics import precision_score, recall_score,f1_score
print(precision_score(tag_te,prd2))
print(recall_score(tag_te,prd2))
print(f1_score(tag_te,prd2))

from sklearn.metrics import roc_auc_score
roc_auc_score(tag_te,model2.predict(X_te3))

model2.summary()

################### VGG16

IMG_SIZE = (224,224)
IMG_SHAPE = IMG_SIZE + (3,)
base_model3 = tf.keras.applications.VGG16(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')

base_model3.trainable = False

tf.keras.utils.set_random_seed(987)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
dense1_layer = tf.keras.layers.Dense(64,activation="elu")
prediction_layer = tf.keras.layers.Dense(1, activation="sigmoid")

inputs = tf.keras.Input(shape=(224, 224, 3))

x = base_model3(inputs, training=False)
x = global_average_layer(x)
x = dense1_layer(x)
outputs = prediction_layer(x)
model3 = tf.keras.Model(inputs, outputs)

from tensorflow.keras.optimizers import RMSprop, Adam
model3.compile(optimizer=Adam(learning_rate=1e-4), loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'])

model3.fit(X_tr,tag_tr,epochs=300,batch_size=100,shuffle=True)

prd = model3.predict(X_te3)
prd3 = []
for i in range(len(tag_te)):
  if prd[i] < 0.5:
    prd3.append(0)
  else:
    prd3.append(1)

from sklearn.metrics import ConfusionMatrixDisplay
dis = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(tag_te,prd3), display_labels=[0,1])
dis.plot()
#plt.grid()
plt.show()

from sklearn.metrics import classification_report
print(classification_report(tag_te,prd3,digits=4))

from sklearn.metrics import precision_score, recall_score,f1_score
print(precision_score(tag_te,prd3))
print(recall_score(tag_te,prd3))
print(f1_score(tag_te,prd3))

from sklearn.metrics import roc_auc_score
roc_auc_score(tag_te,model3.predict(X_te3))

model3.summary()

