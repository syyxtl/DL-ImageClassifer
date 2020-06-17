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
		# reshape layer
		self.model.add(Reshape((28, 28, 1, ), input_shape=(28*28, )))
		# conv 1, 1 input, 8 output
		self.model.add(Convolution2D(8, 3, 3, border_mode='same'))
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2, border_mode='same', data_format='channels_first'))
		# conv 2, 8 input, 16 output
		self.model.add(Convolution2D(16, 3, 3, border_mode='same'))
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2, border_mode='same', data_format='channels_first'))
		# full connected layer
		self.model.add(Flatten())
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