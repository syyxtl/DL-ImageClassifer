#! /usr/bin/env python


import os
import csv
import time
import datetime
import numpy as np

import inputs
from _cnn import CNN

import tensorflow as tf
from tensorflow.contrib import learn

# Parameters
# ==================================================
# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1592286110/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

class TextCNN_:  
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'    #use GPU with ID=0
        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
        self.config.gpu_options.allow_growth = True #allocate dynamically
        self.sess = tf.Session(config = self.config)

    def doevalcnn(self, images, labels= None):
        x_test = images
        y_test = labels
        print("\nEvaluating...\n")
        print(FLAGS.checkpoint_dir)
        # Evaluation
        # ==================================================
        checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(self.sess, checkpoint_file)
                # Get the placeholders from the graph by name
                inputX = graph.get_operation_by_name("inputX").outputs[0]
                # or  inputX = graph.get_tensor_by_name("inputX:0")
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
                # Tensors we want to evaluate
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]
                # Generate batches for one epoch
                batches = inputs.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
                # Collect the predictions here
                all_predictions = []
                for x_test_batch in batches:
                    batch_predictions = self.sess.run(predictions, {inputX: x_test_batch, dropout_keep_prob: 1.0})
                    all_predictions = np.concatenate([all_predictions, batch_predictions])
                    print(batch_predictions)
                
                t_prediction = self.sess.run(tf.argmax(y_test,1))
                print(t_prediction)

if __name__ == '__main__':
    tcnn = TextCNN_()
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    eval_x = mnist.train.images[0:50]
    eval_y = mnist.train.labels[0:50]
    tcnn.doevalcnn(eval_x, eval_y)
    # tcnn.dotextcnn("this is a function and this is not a function")