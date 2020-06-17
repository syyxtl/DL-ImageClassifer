from keras.models import Sequential
from keras.layers import Reshape, Convolution2D, Activation, MaxPooling2D, Flatten, Dense

from keras.optimizers import SGD

class Inputs(object):
	def __init__(self):
		from tensorflow.examples.tutorials.mnist import input_data
		self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

	def load_train_data_and_labels(self):
		train_x = self.mnist.train.images
		train_y = self.mnist.train.labels
		return train_x, train_y

	def load_eval_data_and_labels(self):
		eval_x = self.mnist.test.images
		eval_y = self.mnist.test.labels
		return eval_x, eval_y

class CNN(object):
	def __init__(self):
		self.model = Sequential()
		self.model.add(Dense(512,activation='relu',input_shape=(784,)))
		self.model.add(Dense(256,activation='relu'))
		self.model.add(Dense(10,activation='softmax'))
		self.model.summary()

	def back_model(self):
		return self.model

##### train #####
inputs = Inputs()
train_x, train_y = inputs.load_train_data_and_labels()
eval_x, eval_y = inputs.load_eval_data_and_labels()

model = CNN().back_model()
model.compile(optimizer=SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=64, epochs=5, validation_split=0.1)
score = model.evaluate(eval_x, eval_y)
print("loss:",score[0])
print("accu:",score[1])

