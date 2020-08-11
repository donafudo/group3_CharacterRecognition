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

    cnn = cnn.cnn()
    
    cnn.predict(image)

    print(image.shape)
    