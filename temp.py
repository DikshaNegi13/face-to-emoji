import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import PIL
from tensorflow.keras import layers
import tensorflow as tf
import pickle
from tensorflow.python.keras import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from IPython.display import display
from tensorflow.keras import backend as K
from keras.utils.np_utils import to_categorical 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random


def res_block(X, filter, stage):

        # Convolutional_block
        X_copy = X

        f1 , f2, f3 = filter

        # Main Path
        X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_conv_a', kernel_initializer= glorot_uniform(seed = 0))(X)
        X = MaxPool2D((2,2))(X)
        X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_a')(X)
        X = Activation('relu')(X) 

        X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_conv_b', kernel_initializer= glorot_uniform(seed = 0))(X)
        X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_b')(X)
        X = Activation('relu')(X) 

        X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_conv_c', kernel_initializer= glorot_uniform(seed = 0))(X)
        X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_c')(X)


        # Short path
        X_copy = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_conv_copy', kernel_initializer= glorot_uniform(seed = 0))(X_copy)
        X_copy = MaxPool2D((2,2))(X_copy)
        X_copy = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_copy')(X_copy)

        # ADD
        X = Add()([X,X_copy])
        X = Activation('relu')(X)

        # Identity Block 1
        X_copy = X


        # Main Path
        X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_identity_1_a', kernel_initializer= glorot_uniform(seed = 0))(X)
        X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_a')(X)
        X = Activation('relu')(X) 

        X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_identity_1_b', kernel_initializer= glorot_uniform(seed = 0))(X)
        X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_b')(X)
        X = Activation('relu')(X) 

        X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_identity_1_c', kernel_initializer= glorot_uniform(seed = 0))(X)
        X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_c')(X)

        # ADD
        X = Add()([X,X_copy])
        X = Activation('relu')(X)

        # Identity Block 2
        X_copy = X


        # Main Path
        X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_identity_2_a', kernel_initializer= glorot_uniform(seed = 0))(X)
        X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_a')(X)
        X = Activation('relu')(X) 

        X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_identity_2_b', kernel_initializer= glorot_uniform(seed = 0))(X)
        X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_b')(X)
        X = Activation('relu')(X) 

        X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_identity_2_c', kernel_initializer= glorot_uniform(seed = 0))(X)
        X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_c')(X)

        # ADD
        X = Add()([X,X_copy])
        X = Activation('relu')(X)

        return X


 
input_shape = (48, 48, 1)

# Input tensor shape
X_input = Input(input_shape)

# Zero-padding
X = ZeroPadding2D((3, 3))(X_input)

# 1 - stage
X = Conv2D(64, (7, 7), strides= (2, 2), name = 'conv1', kernel_initializer= glorot_uniform(seed = 0))(X)
X = BatchNormalization(axis =3, name = 'bn_conv1')(X)
X = Activation('relu')(X)
X = MaxPooling2D((3, 3), strides= (2, 2))(X)

# 2 - stage
X = res_block(X, filter= [64, 64, 256], stage= 2)

# 3 - stage
X = res_block(X, filter= [128, 128, 512], stage= 3)

# Average Pooling
X = AveragePooling2D((2, 2), name = 'Averagea_Pooling')(X)

# Final layer
X = Flatten()(X)
X = Dense(7, activation = 'softmax', name = 'Dense_final', kernel_initializer= glorot_uniform(seed=0))(X)

model_emotion = Model( inputs= X_input, outputs = X)




model_emotion.load_weights('FacialExpression_weights1.hdf5')

emotion_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}



emoji_dist={0:"emojis/angry.png",2:"emojis/disgusted.png",2:"emojis/fearful.png",3:"emojis/happy.png",4:"emojis/neutral.png",5:"emojis/sad.png",6:"emojis/surprised.png"}


last_frame=np.zeros((480, 640, 3), dtype=np.uint8)

def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = equalize(img)
    img = img/255
    return img

frameWidth= 640
frameHeight = 480
brightness = 180
threshold = 0.5
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while(True):
# Capture frame-by-frame
    ret, frame = cap.read()
    # cv2.resize(frame,(600,500))
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 3.5, 7)
    
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]       
        temp=cv2.resize(roi_gray, (48, 48))
        cropped_img = cv2.resize(roi_gray, (48, 48))
        cropped_img = preprocessing(cropped_img)
        cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)
        prediction = model_emotion.predict(cropped_img)
        probabilityValue =np.amax(prediction)
        maxindex = int(np.argmax(prediction))
       
   
        if probabilityValue > threshold:
	        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
	        cv2.putText(frame, str(round(probabilityValue*100,2) )+"%", (180, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        if probabilityValue > threshold:
            imgt=cv2.imread(str(emoji_dist[maxindex]))
            roi=frame[y:y+h, x:x+w]
            img=cv2.resize(imgt,(h,w))
            
           

            img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

            img2_fg = cv2.bitwise_and(img,img,mask = mask)

  
            dst = cv2.add(img1_bg,img2_fg)
            frame[y:y+h,x:x+w]= dst
            
        cv2.imshow('face-to-emoji',frame)
   
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

# When everything done, release the capture
cap.release()
cv2.waitkey(0)
cv2.destroyAllWindows()

    



