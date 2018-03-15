import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

import dataset

MODEL_DIR = ".\\.tmp"
BATCH_SIZE = 100

def inference(features):
	# Layer #0: Input 16x16x1
	input_layer = tf.reshape(
		tensor = features,
		shape = [-1, dataset.FEATURE_WIDTH, dataset.FEATURE_HEIGHT, 1]
	)
	
	# Layer #1: Conv 4x4x128
	conv1 = tf.layers.conv2d(
		inputs = input_layer,
		filters = 128,
		kernel_size = [4, 4],
		padding = "same",
		activation = tf.nn.relu,
		kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1.0),
		name = "conv1"
	)
	pool1 = tf.layers.max_pooling2d(
		inputs = conv1,
		pool_size = [4, 4],
		strides = 4,
		name = "pool1"
	)

	# Layer #2: Conv 2x2x256
	conv2 = tf.layers.conv2d(
		inputs = pool1,
		filters = 256,
		kernel_size = [2, 2],
		padding = "same",
		activation = tf.nn.relu,
		kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 1.0),
		name = "conv2"
	)
	pool2 = tf.layers.max_pooling2d(
		inputs = conv2,
		pool_size = [2, 2],
		strides = 2,
		name = "pool2"
	)

	# Layer #3: Concat 1024
	pool2_flat = tf.reshape(
		tensor = pool2,
		shape = [-1, 1024],  #[-1, tf.cast(tf.divide(tf.size(pool2), tf.shape(pool2)[0]), tf.int32)]
		name = "pool2_flat"
	)
	
	# Layer #4: Dense 256
	dense1 = tf.layers.dense(
		inputs = pool2_flat,
		units = 256,
		activation = tf.nn.relu,
		name = "dense1"
	)

	# Layer #5: Dense 32
	dense2 = tf.layers.dense(
		inputs = dense1,
		units = 32,
		activation = tf.nn.relu,
		name = "dense2"
	)

	# Layers #6: Logits 2
	logits = tf.layers.dense(
		inputs = dense2,
		units = 2,
		activation = tf.nn.relu,
		name = "logits"
	)

	# Layers #7: Softmax 2
	softmax = tf.nn.softmax(
		logits = logits,
		name = "softmax_tensor"
	)
	return logits, softmax

def train(labels, logits):
	
	loss = tf.nn.softmax_cross_entropy_with_logits_v2(
		labels = labels,
		logits = logits
	)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
	optimizer.minimize(loss)
	return labels, logits, loss

def main(unused_argv):
	iterator = dataset.get().make_initializable_iterator()
	features, labels = iterator.get_next()
	onehot_labels = tf.one_hot(indices = labels, depth = 2)

	logits, softmax = inference(features)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(iterator.initializer)
		try:
			for i in range (1000):
				print(sess.run(
					train(onehot_labels, logits)
				))
		except tf.errors.OutOfRangeError:
			print("END!")

if __name__ == "__main__":
	tf.app.run()
