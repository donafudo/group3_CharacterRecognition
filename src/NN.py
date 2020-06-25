from keras import backend as K
from keras.datasets import mnist
import keras
num_classes=10
#[参照] 「ETL Character Database」
(x_train, y_train), (x_test, y_test)  = mnist.load_data()

#x_train = x_train.reshape(60000, 784) # 2次元配列を1次元に変換
#x_test = x_test.reshape(10000, 784)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')   # int型をfloat32型に変換
x_test = x_test.astype('float32')
x_train /= 255                        # [0-255]の値を[0.0-1.0]に変換
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
#model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy'])
model.fit(
    x_train, y_train,
    batch_size=100, epochs=20,
    validation_data=(x_test, y_test))

model.save("./model_001")




