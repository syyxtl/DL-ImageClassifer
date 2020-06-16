import tensorflow as tf
import numpy as np


class CNN(object):
	'''
		common CNN structure 
	'''
	def __init__(self, graphh, graphw, num_classes, l2_reg_lambda=0.0):
		# Placeholders for input, output and dropout
		self.inputX = tf.placeholder(tf.float32, [None, (graphh*graphw)], name="inputX")
		self.inputY = tf.placeholder(tf.float32, [None, num_classes], name="inputY")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

		# [w,h,in,out] #[长, 宽, 高in, 高out]
		self.weights = {
			'wc1':tf.Variable(tf.truncated_normal([5,5,1,8], stddev=0.1)),
			'wc2':tf.Variable(tf.truncated_normal([5,5,8,16], stddev=0.1)),
			'wd1':tf.Variable(tf.truncated_normal([7*7*16,64], stddev=0.1)),
			'out':tf.Variable(tf.truncated_normal([64,num_classes], stddev=0.1)),
		}

		self.biases = {
			'bc1':tf.Variable(tf.truncated_normal([8], stddev=0.1)),
			'bc2':tf.Variable(tf.truncated_normal([16], stddev=0.1)),
			'bd1':tf.Variable(tf.truncated_normal([64], stddev=0.1)),
			'out':tf.Variable(tf.truncated_normal([num_classes], stddev=0.1)),
		}

		with tf.name_scope("changex"):
			Xin = tf.reshape(self.inputX, [-1, graphh, graphw, 1])

		# Create a convolution + maxpool layer for each filter size
		with tf.name_scope("conv-maxpool"):
			# conv1 5*5 conv,1 input ,8 output
			convly = Xin
			convly = tf.nn.conv2d(convly, self.weights['wc1'], strides=[1,1,1,1], padding="SAME", dilations=None, name='conv_1')
			convly = tf.nn.bias_add(convly, self.biases['bc1'], name='bias_1')
			convly = tf.nn.relu(convly, name='activate_1')
			convly = tf.nn.max_pool(convly, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME", name='pooling_1')
			# conv2 5*5 conv ,8 input, 16 output
			conv2y = tf.nn.conv2d(convly, self.weights['wc2'], strides=[1,1,1,1], padding="SAME", dilations=None, name='conv_2')
			conv2y = tf.nn.bias_add(conv2y, self.biases['bc2'], name='bias_2')
			conv2y = tf.nn.relu(conv2y, name='activate_2')
			conv2y = tf.nn.max_pool(conv2y, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME", name='pooling_2')

		with tf.name_scope("full-connect"):
			fullc1 = tf.reshape(conv2y, [-1,self.weights['wd1'].get_shape().as_list()[0]] )
			fullc1 = tf.nn.xw_plus_b(fullc1, self.weights['wd1'], self.biases['bd1'])
			fullc1 = tf.nn.relu(fullc1)
			fullc1 = tf.nn.dropout(fullc1, self.dropout_keep_prob)

		l2_loss = tf.constant(0.0)
		with tf.name_scope("output"):
			l2_loss += tf.nn.l2_loss(self.weights['out'])
			l2_loss += tf.nn.l2_loss(self.biases['out'])
			self.scores = tf.nn.xw_plus_b(fullc1, self.weights['out'], self.biases['out'], name="scores")
			self.predictions = tf.argmax(self.scores, 1, name="predictions") # argmax -> x,axis=0列(axis=1行)

		# Calculate mean cross-entropy loss
		with tf.name_scope("loss"):
			losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.inputY)
			self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
		with tf.name_scope("accuracy"):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.inputY, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


		# conv = Xin
		# # Create a convolution + maxpool layer for each filter size
		# for i, filter_size in enumerate(filter_sizes):
		# 	with tf.name_scope("conv-maxpool-%s" % i):
		# 		convly = tf.nn.conv2d(input=Xin, filters=self.weights['wc1'], strides=[1,1,1,1], padding="SAME", dilations=None, name='conv')
		# 		convly = tf.nn.bias_add(convly, self.biases['bc1'], name='bias')
		# 		convly = tf.nn.relu(convly, name='activate')
		# 		convly = tf.nn.max_pool2d(convly, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME", name='pooling')

		
