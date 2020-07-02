from keras import backend as K
from keras.datasets import mnist
import keras
import pandas as pd
import LoadETL

num_classes=75
#[参照] 「ETL Character Database」
(x_train, y_train), (x_test, y_test)  = LoadETL.load_data()
img_size = 40

x_train = x_train.reshape(x_train.shape[0], img_size, img_size, 1)
x_test = x_test.reshape(x_test.shape[0], img_size, img_size, 1)

#labelをone hot vectorに変換
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

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

model.save("./model_001")
