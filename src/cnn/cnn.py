import os
import tensorflow as tf
import numpy as np

import import_data

SAMPLE_LENGTH = 2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def cnn_model(features, lables, mode):
	input_layer = features
	return input_layer

dataset = import_data.get_dataset()
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()  

with tf.Session() as sess:
	sess.run(iterator.initializer)
	try:
		for i in range (SAMPLE_LENGTH):
			print(sess.run(next_element))
	except tf.errors.OutOfRangeError:
		print("END!")