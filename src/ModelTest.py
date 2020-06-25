from PIL import Image
import PIL.ImageOps
import numpy as np
import matplotlib.pyplot as plt
import keras

model = keras.models.load_model("./model_001")

size = 28
 
for num in range(10):

    test_image = "./testImgs/{}.jpg".format(num)
    
    image = Image.open(test_image).convert("L")

    image = image.resize((size, size))
    
    #reverse image
    image = PIL.ImageOps.invert(image)
    #make image flat
    in_fact_data = [np.array(image).flatten()]
    #convert to np array
    in_fact_data = np.array(in_fact_data)
    in_fact_data = in_fact_data / 255 
    in_fact_data = np.reshape(in_fact_data,(1,28,28,1))

    plt.show()

    expect = model.predict(in_fact_data)
    print("value:{}".format(num))
    print("predict:{}".format(np.argmax(expect)))
    print(expect[0][np.argmax(expect)])