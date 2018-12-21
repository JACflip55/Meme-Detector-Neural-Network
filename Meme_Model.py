import tensorflow as tf
from tflearn.data_utils import image_preloader
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import numpy as np
import h5py

tflearn.config.init_graph(soft_placement=True)

dataset = h5py.File('dataset.h5','r')
X = dataset['X']
Y = dataset['Y']

#Split dataset: 80% training, 10% validation and 10% testing
validation_X = X[-(len(X) // 10):]
validation_Y = Y[-(len(X) // 10):]

test_X = X[-(len(X) // 5):-(len(X) // 10)]
test_Y = Y[-(len(X) // 5):-(len(X) // 10)]

X = X[:-(len(X) // 5)]
Y = Y[:-(len(X) // 5)]

convnet = input_data(shape=[None, 128, 128,3], name='input')

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 32, activation='relu')
convnet = dropout(convnet, 0.5)

convnet = fully_connected(convnet, 16, activation='relu')
convnet = dropout(convnet, 0.5)

#2 classes meme and non-meme
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.005, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)
model.load('meme_model.model')
with tf.device("/gpu:0"):
   #Train
   model.fit({'input': X}, {'targets': Y},n_epoch=20,validation_set=({'input': validation_X}, {'targets':validation_Y}), show_metric=True, run_id='meme_classifier',batch_size =16,validation_batch_size=16)
   #Test
   print(model.predict(test_X))

model.save('meme_model.model')