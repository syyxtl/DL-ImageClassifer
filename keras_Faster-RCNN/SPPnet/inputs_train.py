import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from keras.preprocessing import image
from keras.models import load_model
from keras.optimizers import SGD
import SPPnet

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


def load_valid_data_and_labels(dataSet= "./validation"):
	images = []
	imagespaths = [str(dataSet + "/" + jpg) for jpg in os.listdir(dataSet)]
	labels = [jpg.split("_")[0]  for jpg in os.listdir(dataSet)]

	for imagePath in imagespaths:
		image = cv2.imread(imagePath)
		images.append(image)

	labels = np.array(LabelBinarizer().fit_transform(labels) )
	images = np.array(images)
	return images, labels


## train
train_x, train_y = load_train_data_and_labels()
model = SPPnet.SPPNET([1, 2, 4]).back_model()
model.compile(optimizer=SGD(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=1, epochs=20, validation_split=0.1)
model.save("food.h5")