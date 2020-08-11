import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import pandas as pd
import numpy as np
import load_etl
import cv2

class cnn:
    """
    Create CNN model

    Attributes:
    ------------
    img_size : int
        input image size
    model_path : string
        Save and load destination path
    """

    def __init__(self):
        self.img_size=40
        self.model_path="./model_001"

    def training(self):
        """
        training CNN
        """

        num_classes=75
        #[参照] 「ETL Character Database」
        (x_train, y_train), (x_test, y_test)  = LoadETL.load_data()

        x_train = x_train.reshape(x_train.shape[0], img_size, img_size, 1)
        x_test = x_test.reshape(x_test.shape[0], img_size, img_size, 1)

        #labelをone hot vectorに変換
        y_train = pd.get_dummies(y_train)
        y_test = pd.get_dummies(y_test)

        #modeの構築
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_size, img_size, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(75, activation='softmax'))
        model.summary()

        model.compile(
            loss='categorical_crossentropy',
            optimizer='sgd',
            metrics=['accuracy'])
        model.fit(
            x_train, y_train,
            batch_size=50, epochs=50,
            validation_data=(x_test, y_test))

        model.save(self.model_path)
    
    def predict(self, image):
        """
        Predict character image 

        Parameters:
        ------------
        image : numpy.ndarray
            shape(img_width, img_height)
        """
        
        ret, image = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY)
        #resize image
        image = cv2.resize(image, (self.img_size, self.img_size))
        #binarization
        ret, image = cv2.threshold(image, 250, 255, cv2.THRESH_BINARY)
        #reverse image
        image = cv2.bitwise_not(image)
        #make image flat
        in_fact_data = [np.array(image).flatten()]
        #convert to np array
        in_fact_data = np.array(in_fact_data)
        in_fact_data = in_fact_data / 255 
        in_fact_data = np.reshape(in_fact_data,(1,self.img_size,self.img_size,1))

        #予測
        model = keras.models.load_model(self.model_path)
        expect = model.predict(in_fact_data)

        #index番号からラベル名(ひらがな)に変換するためのリスト
        s=pd.Series(list("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽっゃゅょ"))
        labels=pd.get_dummies(s).columns

        #確率の高い三つの候補を表示
        args=expect[0].argsort()[::-1]
        for i in args[:3]:
            print("predict:{}".format(labels[i]))
            print(expect[0][i])
            print()
