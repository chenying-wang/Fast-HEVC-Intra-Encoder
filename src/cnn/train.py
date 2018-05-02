import sys
import tensorflow as tf
import numpy as np

import data
import cnn
import new_cnn

EPOCH = 1
SAMPLE_SIZE = 51520000
BATCH_SIZE = 16

INIT_LEARNING_RATE = 0.001
DECAY_STEPS = 50000
DECAY_RATE = 0.98
KEEP_PROB = 0.6

CKPT_PATH = lambda size_index: "./.tmp/" + str(size_index) + "/"
CKPT_PREFIX = lambda size_index: "training_" + str(size_index) + ".ckpt"
CKPT_STEP = 1000
GRAPH_FILENAME = lambda size_index: "cnn_modle_" + str(size_index) + ".pbtxt"
TRAINING_LOG_DIR = lambda size_index: CKPT_PATH(size_index) + "train_" + str(size_index) + "/"

def train(size_index):

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

	with tf.variable_scope("train") as scope:
		global_step = tf.Variable(initial_value = 1, trainable = False, name = "global_step")
		pred = tf.argmax(logits, 1)

		onehot_lables = tf.one_hot(indices = labels, depth = 2, dtype = tf.int64)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
			labels = onehot_lables,
			logits = logits
		))
		
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

	with tf.Session() as sess:
		saver = tf.train.Saver()
		tf.global_variables_initializer().run()
		iterators.initializer.run()

		tf.train.write_graph(sess.graph, CKPT_PATH(size_index), GRAPH_FILENAME(size_index))

		if tf.train.get_checkpoint_state(CKPT_PATH(size_index)):
			saver.restore(sess, tf.train.latest_checkpoint(CKPT_PATH(size_index)))

		for _ in range (int(EPOCH * SAMPLE_SIZE / BATCH_SIZE)):
			_global_step = sess.run(global_step)

			_, _loss, _accuracy,  = sess.run(
				[train_batch, loss, accuracy]
			)
			print("Step %d, loss = %f, accuracy = %.2f%%, width = height = %d" % (_global_step, _loss, 100 * _accuracy, data.FEATURE_WIDTH[size_index]))
			summary = sess.run(merged)
			summary_writer.add_summary(summary, _global_step)
			
			if _global_step % CKPT_STEP == 0:
				saver.save(
					sess = sess,
					save_path = CKPT_PATH(size_index) + CKPT_PREFIX(size_index),
					global_step = _global_step
				)
				print("Model Saved")
		
		print("Training Done")

	summary_writer.close()

def main(unused_argv):
	train(size_index = int(sys.argv[1]))

if __name__ == "__main__":
	tf.app.run()
 