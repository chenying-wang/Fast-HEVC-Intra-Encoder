import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

import dataset

MODEL_DIR = ".\\.tmp"
EPOCH = 3
SAMPLE_SIZE = 2304000
SAMPLE_SIZE = 480
BATCH_SIZE = 16

def inference(features):
	# Layer #0: Input 16x16x1
	input_layer = tf.reshape(
		tensor = features,
		shape = [BATCH_SIZE, dataset.FEATURE_WIDTH, dataset.FEATURE_HEIGHT, 1]
	)
	
	# Layer #1: Conv 4x4x128
	with tf.variable_scope("conv1") as scope:
		kernel = tf.get_variable(
			name = "weights",
			shape = [4, 4, 1, 128],
			dtype = tf.float32,
			initializer = tf.random_normal_initializer(mean = 0.0, stddev = 2 / 256)
		)
		conv = tf.nn.conv2d(
			input = input_layer,
			filter = kernel,
			strides = [1, 1, 1, 1],
			padding = "SAME"
		)
		biases = tf.get_variable(
			name = "biases",
			shape = [128],
			dtype = tf.float32,
			initializer = tf.zeros_initializer()
		)
		pre_activation = tf.nn.bias_add(
			value = conv,
			bias = biases
		)
		conv1 = tf.nn.relu(
			features = pre_activation,
			name = scope.name
		)
	with tf.variable_scope("pool1") as scope:
		pool1 = tf.nn.max_pool(
			value = conv1,
			ksize = [1, 4, 4, 1],
			strides = [1, 4, 4, 1],
			padding = "SAME",
			name = scope.name
		)

	# Layer #2: Conv 2x2x256
	with tf.variable_scope("conv2") as scope:
		kernel = tf.get_variable(
			name = "weights",
			shape = [2, 2, 128, 256],
			dtype = tf.float32,
			initializer = tf.random_normal_initializer(mean = 0.0, stddev = 2 / 2048)
		)
		conv = tf.nn.conv2d(
			input = pool1,
			filter = kernel,
			strides = [1, 1, 1, 1],
			padding = "SAME"
		)
		biases = tf.get_variable(
			name = "biases",
			shape = [256],
			dtype = tf.float32,
			initializer = tf.zeros_initializer()
		)
		pre_activation = tf.nn.bias_add(
			value = conv,
			bias = biases
		)
		conv2 = tf.nn.relu(
			features = pre_activation,
			name = scope.name
		)
	with tf.variable_scope("pool2") as scope:
		pool2 = tf.nn.max_pool(
			value = conv2,
			ksize = [1, 2, 2, 1],
			strides = [1, 2, 2, 1],
			padding = "SAME",
			name = scope.name
		)

	# Layer #3: Concat 1024
	with tf.variable_scope("concat") as scope:
		pool2_flat = tf.reshape(
			tensor = pool2,
			shape = [-1, 1024],  #[-1, tf.cast(tf.divide(tf.size(pool2), tf.shape(pool2)[0]), tf.int32)]
			name = "pool2_flat"
		)
	
	# Layer #4: Dense 256
	with tf.variable_scope("dense1") as scope:
		weights = tf.get_variable(
			name = "weights",
			shape = [1024, 256],
			dtype = tf.float32,
			initializer = tf.random_normal_initializer(mean = 0.0, stddev = 2 / 1024)
		)
		biases = tf.get_variable(
			name = 'biases',
			shape = [256],
			initializer = tf.zeros_initializer()
		)
		dense1 = tf.nn.relu(
			features = tf.matmul(pool2_flat, weights) + biases,
			name = scope.name
		)

	# Layer #5: Dense 64
	with tf.variable_scope("dense2") as scope:
		weights = tf.get_variable(
			name = "weights",
			shape = [256, 64],
			dtype = tf.float32,
			initializer = tf.random_normal_initializer(mean = 0.0, stddev = 2 / 256)
		)
		biases = tf.get_variable(
			name = 'biases',
			shape = [64],
			initializer = tf.zeros_initializer()
		)
		dense2 = tf.nn.relu(
			features = tf.matmul(dense1, weights) + biases,
			name = scope.name
		)

	# Layers #6: Logits 2
	with tf.variable_scope("logits") as scope:
		weights = tf.get_variable(
			name = "weights",
			shape = [64, 2],
			dtype = tf.float32,
			initializer = tf.random_normal_initializer(mean = 0.0, stddev = 2 / 64)
		)
		biases = tf.get_variable(
			name = 'biases',
			shape = [2],
			initializer = tf.zeros_initializer()
		)
		logits = tf.nn.relu(
			features = tf.matmul(dense2, weights) + biases,
			name = scope.name
		)

	# Layers #7: Softmax 2
	with tf.variable_scope("softmax") as scope:
		softmax = tf.nn.softmax(
			logits = logits,
			name = scope.name
		)

	return logits, softmax
	
def train():
	with tf.variable_scope("input") as scope:
		iterator = dataset.get().batch(BATCH_SIZE).make_initializable_iterator()
		features, labels = iterator.get_next()
		onehot_labels = tf.one_hot(indices = labels, depth = 2, dtype = tf.int64)

	logits, softmax = inference(features)
	pred = tf.argmax(logits, 1)

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
		labels = onehot_labels,
		logits = logits
	))
	
	accuracy = tf.reduce_mean(tf.cast(
		tf.equal(pred, labels)
	, tf.float32))

	optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
	train_batch = optimizer.minimize(loss)
	
	tf.summary.scalar("loss", loss)
	tf.summary.scalar("accuracy", accuracy)
	merged = tf.summary.merge_all()	

	summary_writer = tf.summary.FileWriter(MODEL_DIR, tf.get_default_graph())

	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		iterator.initializer.run()
		try:
			for step in range (int(EPOCH * SAMPLE_SIZE / BATCH_SIZE)):
				_, avg_loss = sess.run([train_batch, loss])
				summary = sess.run(merged)
				summary_writer.add_summary(summary, step)
				print("Step %d, loss = %f" % (step, avg_loss))
		except tf.errors.OutOfRangeError:
			print("END!")
		
	summary_writer.close()

def main(unused_argv):
	train()

if __name__ == "__main__":
	tf.app.run()
