from keras.models import Sequential
from keras.layers import Reshape, Convolution2D, Activation, MaxPooling2D, Flatten, Dropout, Dense 
from keras.layers import LeakyReLU
from SpatialPyramidPooling import SpatialPyramidPooling

class SPPNET(object):
	def __init__(self, bin):
		self.model = Sequential()
		# reshape layer
		# None
		# conv 1, 1 input, 8 output
		self.model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(None, None, 3 )))
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2, border_mode='same', data_format='channels_last'))
		# conv 2, 8 input, 16 output
		self.model.add(Convolution2D(32, 3, 3, border_mode='same'))
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2, border_mode='same', data_format='channels_last'))
		# conv 3, 16 input, 32 output
		self.model.add(Convolution2D(64, 3, 3, border_mode='same'))
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2, border_mode='same', data_format='channels_last'))
		# full connected layer
		self.model.add(SpatialPyramidPooling(bin))
		# dropout
		# None
		self.model.add(Dense(11,activation='softmax'))
		self.model.summary()

	def back_model(self):
		return self.model