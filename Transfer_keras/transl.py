import os
import cv2
import numpy as np
from keras.applications.vgg16 import VGG16
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from keras.layers import MaxPooling2D, Dense, Flatten , GlobalAveragePooling2D
from keras.preprocessing import image
from keras.optimizers import SGD
from keras.models import load_model
from keras.models import Model

def load_train_data_and_labels(dataSet= "./training"):
	images = []
	imagespaths = [str(dataSet + "/" + jpg) for jpg in os.listdir(dataSet)]
	labels = [jpg.split("_")[0]  for jpg in os.listdir(dataSet)]

	for imagePath in imagespaths:
		image = cv2.imread(imagePath)
		image = cv2.resize(image, (128, 128))
		images.append(image)

	labels = np.array(LabelBinarizer().fit_transform(labels) )
	images = np.array(images)
	return images, labels

def add_new_layer(base_model): # layer 19 - 22
	x = base_model.output
	x = MaxPooling2D()(x)
	x = Flatten()(x)
	# x = GlobalAveragePooling2D()(x)
	x = Dense(128, activation='relu')(x)
	predictions = Dense(11, activation='softmax')(x)
	return predictions

# create the base pre-trained model #defalut input 224 * 224 
base_model = VGG16(weights='imagenet', input_shape=(128, 128, 3), include_top=False) # 18 layers

# add new layer for TL
new_model = add_new_layer(base_model)

# the latest model
model = Model(inputs=base_model.input, outputs=new_model)

for layer in model.layers[:19]:
   layer.trainable = False
for layer in model.layers[19:]:
   layer.trainable = True

train_x, train_y = load_train_data_and_labels()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=8, epochs=20, validation_split=0.1)