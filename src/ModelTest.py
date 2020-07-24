from PIL import Image
import PIL.ImageOps
import numpy as np
import pandas as pd
import keras
import cv2
import sys
import os

model = keras.models.load_model("./model_001")

size = 40
 
 #index番号からラベル名(ひらがな)に変換するためのリスト
s=pd.Series(list("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽっゃゅょ"))
labels=pd.get_dummies(s).columns

for img_path in sys.argv[1:]:

    if(os.path.isfile(img_path)==False):
        try:
            raise FileNotFoundError('no such file')
        except FileNotFoundError:
            raise
    
    image = cv2.imread(img_path, 0)
    
    ret, image = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY)
    #resize image
    image = cv2.resize(image, (size, size))
    #binarization
    ret, image = cv2.threshold(image, 250, 255, cv2.THRESH_BINARY)
    #reverse image
    image = cv2.bitwise_not(image)
    #make image flat
    in_fact_data = [np.array(image).flatten()]
    #convert to np array
    in_fact_data = np.array(in_fact_data)
    in_fact_data = in_fact_data / 255 
    in_fact_data = np.reshape(in_fact_data,(1,size,size,1))

    expect = model.predict(in_fact_data)
    args=expect[0].argsort()[::-1]
    for i in args[:3]:
        print("predict:{}".format(labels[i]))
        print(expect[0][i])
        print()