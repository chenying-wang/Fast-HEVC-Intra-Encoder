import sys
import tensorflow as tf
import numpy as np

import data
import cnn
import new_cnn

EPOCH = 1
SAMPLE_SIZE = 51520000
BATCH_SIZE = 32

INIT_LEARNING_RATE = 0.0001
DECAY_STEPS = 5000
DECAY_RATE = 0.94
KEEP_PROB = 0.5

CKPT_PATH = lambda size_index: "./.tmp/" + str(size_index) + "/"
CKPT_PREFIX = "training.ckpt"
CKPT_STEP = 1000
GRAPH_FILENAME = "cnn_modle.pbtxt"
TRAINING_LOG_DIR = lambda size_index: CKPT_PATH(size_index) + "train/"

INIT_LOSS_WEIGHTS = [
	[1.7, 0.3],
	[1.3, 0.7],
	[0.7, 1.3]
]

LOSS_WEIGHTS_STEP = 1000

def train(size_index):
	loss_weights_0 = tf.placeholder(tf.float32, shape = (), name = "loss_weights_0")
	loss_weights_1 = tf.placeholder(tf.float32, shape = (), name = "loss_weights_1")

	with tf.variable_scope("input") as scope:
		iterators = data.Dataset(size_index).get_train().repeat(EPOCH).batch(BATCH_SIZE).make_initializable_iterator()
		raw_features, labels = iterators.get_next()

		raw_features = tf.identity(raw_features, name = "raw_features")
		
		features = tf.reshape(
			tensor = raw_features,
			shape = [-1, data.FEATURE_HEIGHT[size_index], data.FEATURE_WIDTH[size_index], 1],
			name = "features"
		)
	
	logits, softmax = new_cnn.infer(features, size_index, KEEP_PROB)
	pred = tf.argmax(logits, 1)
	
	with tf.variable_scope("train") as scope:
		global_step = tf.Variable(initial_value = 1, trainable = False, name = "global_step")
		pred = tf.argmax(logits, 1)

		# onehot_lables = tf.one_hot(indices = labels, depth = 2, dtype = tf.int64)
		# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
		# 	labels = onehot_lables,
		# 	logits = logits
		# ))

		loss_weights = tf.multiply(
			tf.subtract(loss_weights_1, loss_weights_0),
			tf.cast(labels, tf.float32)
		)
		loss_weights = tf.add(loss_weights, loss_weights_0)
		loss = tf.losses.sparse_softmax_cross_entropy(
			labels = labels,
			logits = logits,
			weights = loss_weights
		)
		
		accuracy = tf.reduce_mean(tf.cast(
			tf.equal(pred, labels)
		, tf.float32))


		learning_rate = tf.train.exponential_decay(
			learning_rate = INIT_LEARNING_RATE,
			decay_steps = DECAY_STEPS,
			decay_rate = DECAY_RATE,
			global_step = global_step,

		)
		optimizer = tf.train.AdamOptimizer(
			learning_rate = learning_rate,
			beta1 = 0.9,
			beta2 = 0.999
		)
		train_batch = optimizer.minimize(loss, global_step = global_step)

		tf.summary.scalar("loss", loss, family = "train")
		tf.summary.scalar("accuracy", accuracy, family = "train")
		merged = tf.summary.merge_all()
		summary_writer = tf.summary.FileWriter(TRAINING_LOG_DIR(size_index), tf.get_default_graph())

	_loss_weights = [INIT_LOSS_WEIGHTS[size_index][0], INIT_LOSS_WEIGHTS[size_index][1]]
	with tf.Session() as sess:
		saver = tf.train.Saver()
		tf.global_variables_initializer().run()
		iterators.initializer.run()

		tf.train.write_graph(sess.graph, CKPT_PATH(size_index), GRAPH_FILENAME)

		if tf.train.get_checkpoint_state(CKPT_PATH(size_index)):
			saver.restore(sess, tf.train.latest_checkpoint(CKPT_PATH(size_index)))

		for _ in range (int(EPOCH * SAMPLE_SIZE / BATCH_SIZE)):
			_global_step = sess.run(global_step)

			_, _loss, _accuracy,  = sess.run(
				[train_batch, loss, accuracy],
				feed_dict = {
					loss_weights_0: _loss_weights[0],
					loss_weights_1: _loss_weights[1]
				}
			)

			print("Step %d, loss = %f, accuracy = %.2f%%, width = height = %d" % (_global_step, _loss, 100 * _accuracy, data.FEATURE_WIDTH[size_index]))
			summary = sess.run(merged,
				feed_dict = {
					loss_weights_0: _loss_weights[0],
					loss_weights_1: _loss_weights[1]
				}
			)
			summary_writer.add_summary(summary, _global_step)
			
			if _global_step % CKPT_STEP == 0:
				saver.save(
					sess = sess,
					save_path = CKPT_PATH(size_index) + CKPT_PREFIX,
					global_step = _global_step
				)
				print("Model Saved")

			if _global_step % LOSS_WEIGHTS_STEP == 0:
				print("Update loss weights")
				_confusion_sum = [[0.0, 0.0],[0.0, 0.0]]
				for idx in range(200):
					_pred, _labels = sess.run(
						[pred, labels]
					)

					for i in range(BATCH_SIZE):
						l = _labels[i]
						p = _pred[i]
						_confusion_sum[l][p] += 1

				_loss_weights[0] = 0.6 * _loss_weights[0] + 0.4 * 2 * (_confusion_sum[1][1] + _confusion_sum[0][1]) / (sum(_confusion_sum[0]) + sum(_confusion_sum[1]))
				_loss_weights[1] = 0.6 * _loss_weights[1] + 0.4 * 2 * (_confusion_sum[0][0] + _confusion_sum[1][0]) / (sum(_confusion_sum[0]) + sum(_confusion_sum[1]))

				print(_confusion_sum)
				print("accuracy = %.2f%%" % (
					100 * (_confusion_sum[0][0] + _confusion_sum[1][1]) / (sum(_confusion_sum[0]) + sum(_confusion_sum[1]))
				))
				print(_loss_weights)

		print("Training Done")

	summary_writer.close()

def main(unused_argv):
	train(size_index = int(sys.argv[1]))

if __name__ == "__main__":
	tf.app.run()
