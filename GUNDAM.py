print("hello python")

!pip install icrawler

#00、SEEDの機体画像収集
from icrawler import crawler
from icrawler.builtin import BingImageCrawler

crawler = BingImageCrawler(storage = {"root_dir": "00"})
crawler.crawl(keyword = "ガンダム00の機体", max_num = 200)

crawler = BingImageCrawler(storage = {"root_dir": "SEED"})
crawler.crawl(keyword = "ガンダムSEEDの機体", max_num=300)

crawler = BingImageCrawler(storage = {"root_dir": "GANDAM"})
crawler.crawl(keyword = "ガンダムの機体", max_num=300)

from PIL import Image
import os, glob
import numpy as np
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

classes = ["00", "SEED","GANDAM"]
num_calsses = len(classes)
image_size = 64
num_testdata = 25

X_train = []
X_test = []
y_train = []
y_test = []

for index, classlabel in enumerate(classes):
  photos_dir = "./" + classlabel
  files = glob.glob(photos_dir + "/*.jpg")
  for i, file in enumerate(files):
    image = Image.open(file)
    image = image.convert("RGB")
    image = image.resize((image_size, image_size))
    data = np.asarray(image)
    if i < num_testdata:
      X_test.append(data)
      y_test.append(index)
    else:
      for angle in range(-20, 20, 5):

        img_r = image.rotate(angle)
        data = np.asarray(img_r)
        X_train.append(data)
        y_train.append(index)

        img_trains = img_r.transpose(Image.FLIP_LEFT_RIGHT)
        data = np.asarray(img_trains)
        X_train.append(data)
        y_train.append(index)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

xy = (X_train, X_test, y_train, y_test)
np.save("./00_SEED.npy", xy)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop

from keras.utils import np_utils
import tensorflow
import keras
import numpy as np

classes = ["00", "SEED","GANDAM"]
num_classes = len(classes)
image_size = 64


#データを読み込む関数

def load_data():
  X_train, X_test, y_train, y_test = np.load("./00_SEED.npy", allow_pickle = True)

  X_train = X_train.astype("float") / 255
  X_test = X_test.astype("float") / 255

  y_train = np_utils.to_categorical(y_train, num_classes)
  y_test = np_utils.to_categorical(y_test, num_classes)

  return X_train, y_train, X_test, y_test

from keras.optimizers import RMSprop

#モデルを学習する関数

def train(X, y, X_test, y_test):
  model = Sequential()


  model.add(Conv2D(32, (3, 3), padding = "same", input_shape = X.shape[1:]))
  model.add(Activation("relu"))
  model.add(Conv2D(32, (3, 3)))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size = (2, 2)))
  model.add(Dropout(0.1))

  model.add(Conv2D(64, (3, 3), padding = "same"))
  model.add(Activation("relu"))
  model.add(Conv2D(64, (3, 3)))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size = (2, 2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(512))
  model.add(Activation("relu"))
  model.add(Dropout(0.45))
  model.add(Dense(3))
  model.add(Activation("softmax"))
  opt = RMSprop(lr = 0.00005, decay = 1e-6)

  model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])
  model.fit(X, y, batch_size = 28, epochs = 100)


  return model



#メイン開発、データの読み込みとモデルの学習を行います。

def main():
    X_train, y_train, X_test, y_test = load_data()
    model = train(X_train, y_train, X_test, y_test)
    model.save("/content/drive/MyDrive/GANDAM.g1")
    

main()  

!weget -0 "001.jpeg" https://www.pakutaso.com/shared/img/thumb/ONeLivFyunikoooon.jpg #URLを入れる

from google.colab import files
uploaded = files.upload()


import keras
import sys, os
import numpy as np
from PIL import Image
from keras.models import load_model

imsize = (64, 64)
                  #↓読み込みたい画像のパスをコピー
testpic = "/content/ONeLivFyunikoooon.jpg"
keras_param = "/content/drive/MyDrive/GANDAM.g1"


def load_image(path):
  img = Image.open(path)
  print(img)
  # img = img.conert("RGB")
  img = img.resize(imsize)
  img = np.asarray(img)
  img = img / 255
  return img 

model = load_model(keras_param)
img = load_image(testpic)
prd = model.predict(np.array([img]))
print(prd)
