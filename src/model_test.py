import cv2
import sys
import os
import cnn



for img_path in sys.argv[1:]:

    if(os.path.isfile(img_path)==False):
        try:
            raise FileNotFoundError('no such file')
        except FileNotFoundError:
            raise
    
    image = cv2.imread(img_path, 0)


    
    #cnn.predict(image)

di_path="./traind_model/"
model_names=["model_20px","model_30px","model_row40px"]
img_sizes=[20,30,40]

#for i in range(1):
cnn_model = cnn.cnn(img_sizes[2],di_path + model_names[2])
cnn_model.training()
