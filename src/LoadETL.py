
import struct
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import PIL.ImageOps
import re

filename = "ETL8G_03"

def read_etl(filename):
    RECORD_SIZE = 8199
    i = 0
    print("Reading {}".format(filename))
    with open("./datasets/ETL8G/"+filename, 'rb') as f:
        while True:
            s = f.read(RECORD_SIZE)
            if s is None or len(s) < RECORD_SIZE:
                break
            r = struct.unpack(">HH8sIBBBBHHHHBB30x8128s11x", s)
            img = Image.frombytes('F', (128, 127), r[14], 'bit', (4, 0))
            img = img.convert('L')
            img = img.point(lambda x: 255 - (x << 4))
            i = i + 1
            dirname = b'\x1b$B' + r[1].to_bytes(2, 'big') + b'\x1b(B'
            dirname = dirname.decode("iso-2022-jp")

            p = re.compile('[\u3041-\u309F]+')
            if p.fullmatch(dirname):
                try:
                    os.makedirs(f"./extract/{dirname}")
                except:
                    pass
                imagefile = f"./extract/{dirname}/{filename}_{i:0>6}.png"
                print(imagefile)
                img.save(imagefile)

#for i in range(1,34):
#    read_etl("ETL8G_{}".format(str(i).zfill(2)))


def load_data():
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]

    test_num=10
    size = 40

    path="./extract/"
    dirs=[d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for dirname in dirs:
        data_temp=[]
        dirpath=path+dirname

        files=[dirpath + "/" + f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))]
        for image_file in files:      
            image = cv2.imread(image_file, 0)

            #reduce noise
            ret, image = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY)
            #resize image
            image = cv2.resize(image, (size, size))
            #binarization
            ret, image = cv2.threshold(image, 250, 255, cv2.THRESH_BINARY)
            #reverse image
            image = cv2.bitwise_not(image)
            #make image flat
            in_fact_data = [np.array(image)]
            #convert to np array
            in_fact_data = np.array(in_fact_data)
            #0-1 normalize
            in_fact_data = in_fact_data / 255 

            in_fact_data=np.squeeze(in_fact_data)
            data_temp.append(in_fact_data)



        #split into training and test data
        np.random.shuffle(data_temp)
        x_train+=data_temp[test_num:]
        x_test+=data_temp[:test_num]

        #add correct label
        for num in range(len(data_temp[test_num:])):
            y_train.append(dirname)
        for num in range(len(data_temp[:test_num])):
            y_test.append(dirname)
    

    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))
