import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

import dataset

MODEL_DIR = ".\\.tmp"
EPOCH = 3
SAMPLE_SIZE = 2304000
BATCH_SIZE = 16

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
		kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 2 / 256),
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
		kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 2 / 2048),
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
		kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 2 / 1024),
		name = "dense1"
	)

	# Layer #5: Dense 64
	dense2 = tf.layers.dense(
		inputs = dense1,
		units = 64,
		activation = tf.nn.relu,
		kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 2 / 256),
		name = "dense2"
	)

	# Layers #6: Logits 2
	logits = tf.layers.dense(
		inputs = dense2,
		units = 2,
		activation = tf.nn.relu,
		kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 2 / 64),
		name = "logits"
	)

	# Layers #7: Softmax 2
	softmax = tf.nn.softmax(
		logits = logits,
		name = "softmax_tensor"
	)
	print(logits)
	return logits, softmax
	
def train():
	with tf.variable_scope("input") as scope:
		iterator = dataset.get().batch(BATCH_SIZE).make_initializable_iterator()
		features, labels = iterator.get_next()
		onehot_labels = tf.one_hot(indices = labels, depth = 2, dtype = tf.int64)

	logits, softmax = inference(features)
	pred = tf.arg_max(logits, 1)

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
