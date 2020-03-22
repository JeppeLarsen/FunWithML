import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import os
import sys

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing import image
import numpy as np

sys.stderr = stderr

img_width, img_height = 224, 224

model = Sequential()
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.load_weights('model_saved.h5')

img = image.load_img(sys.argv[1], target_size=(img_width, img_height))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)

result = model.predict_classes(img)[0][0]
if result == 0:
    out = 'car'
elif result == 1:
    out = 'plane'
print(out)
