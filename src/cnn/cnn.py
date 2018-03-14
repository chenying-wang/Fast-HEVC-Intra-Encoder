import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import import_data

SAMPLE_LENGTH = 1



def cnn_model_fn(features, lables, mode):

	# Layer #0: Input 16x16x1
	input_layer = tf.reshape(
		tensor = features,
		shape = [-1, import_data.FEATURE_WIDTH, import_data.FEATURE_HEIGHT, 1]
	)
	
	# Layer #1: Conv 4x4x128
	conv1 = tf.layers.conv2d(
		inputs = input_layer,
		filters = 128,
		kernel_size = [4, 4],
		padding = "same",
		activation = tf.nn.relu
	)
	pool1 = tf.layers.max_pooling2d(
		inputs = conv1,
		pool_size = [4, 4],
		strides = 4
	)

	# Layer #2: Conv 2x2x256
	conv2 = tf.layers.conv2d(
		inputs = pool1,
		filters = 256,
		kernel_size = [2, 2],
		padding = "same",
		activation = tf.nn.relu
	)
	pool2 = tf.layers.max_pooling2d(
		inputs = conv2,
		pool_size = [2, 2],
		strides = 2
	)

	# Layer #3: Concat 1024
	pool2_flat = tf.reshape(
		tensor = pool2,
		shape = [-1, 1024]  #[-1, tf.cast(tf.divide(tf.size(pool2), tf.shape(pool2)[0]), tf.int32)]
	)
	
	# Layer #4: Dense 256
	dense1 = tf.layers.dense(
		inputs = pool2_flat,
		units = 256,
		activation = tf.nn.relu
	)

	# Layer #5: Dense 32
	dense2 = tf.layers.dense(
		inputs = dense1,
		units = 32,
		activation = tf.nn.relu
	)

	# Layers #6: Logits 2
	logits = tf.layers.dense(
		inputs = dense2,
		units = 2,
		activation = tf.nn.relu
	)

	# Layers #7: Softmax 2
	softmax = tf.nn.softmax(
		logits = logits
	)

	predictions = softmax
	return predictions

dataset = import_data.get_dataset()
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
pred = cnn_model_fn(next_element["features"], next_element["labels"], tf.estimator.ModeKeys.PREDICT)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(iterator.initializer)
	try:
		for i in range (SAMPLE_LENGTH):
			print(sess.run(
				pred
			))
	except tf.errors.OutOfRangeError:
		print("END!")
