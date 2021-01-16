import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2

import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

#veri setinin yüklenmesi
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#veri ön işleme
img_rows = 28
img_cols = 28
x_train = x_train.reshape( x_train.shape[0],img_rows,img_cols,1)
x_test = x_test.reshape( x_test.shape[0],img_rows,img_cols,1)
input_shape = (img_rows,img_cols,1)

#normalizasyon
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /= 255 
x_test /= 255

#One Hot Encoding işlemi
num_classes = 10
y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)
        
#CNN yapısı
model_list = []
score_list = []

batch_size = 128
epochs = 30

filter_numbers = np.array([[16,32,64], [32,64,128], [64,128,256]])

for i in range(3):
    
    print(filter_numbers[i])
    model = Sequential()
    model.add(Conv2D(filter_numbers[i,0], kernel_size = (3,3),activation = "relu",input_shape = input_shape))
    model.add(MaxPooling2D())
    model.add(Conv2D(filter_numbers[i,1], kernel_size = (3,3),activation = "relu"))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(filter_numbers[i,2],activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes,activation = "softmax"))
    
    model.compile(loss = keras.losses.categorical_crossentropy, 
                  optimizer="adam",
                  metrics = ["accuracy"])
    
    history = model.fit(x_train, y_train, 
                        batch_size = batch_size, 
                        epochs = epochs, verbose = 1,
                        validation_data = (x_test,y_test))

    score = model.evaluate(x_test, y_test, verbose = 0)
    print("Model {} Test Loss: {}".format(i+1, score[0]))
    print("Model {} Test Accuracy: {}".format(i+1, score[1]))
    model_list.append(model)
    score_list.append(score)
    
    model.save("modelCNN"+str(i+1)+".h5")
    
    

#modellerin yüklenmesi
modelCNN1 = load_model("modelCNN1.h5")
modelCNN2 = load_model("modelCNN2.h5")
modelCNN3 = load_model("modelCNN3.h5")