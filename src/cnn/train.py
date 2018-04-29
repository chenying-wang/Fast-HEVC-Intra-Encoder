import tensorflow as tf
import numpy as np

import data
import cnn
import new_cnn

TRAINING_LOG_DIR = "./.tmp/train"
EPOCH = 1
SAMPLE_SIZE = 51520000
# SAMPLE_SIZE = 32
BATCH_SIZE = 16

INIT_LEARNING_RATE = 0.001
DECAY_STEPS = 50000
DECAY_RATE = 0.98

KEEP_PROB = 0.6

CKPT_PATH = "./.tmp/"
CKPT_PREFIX = "training.ckpt"
CKPT_STEP = 1000
GRAPH_FILENAME = "cnn_modle.pbtxt"

CU_DEPTH = 2

def train(size_index):
	global_step = tf.Variable(initial_value = 1, trainable = False, name = "global_step")

	with tf.variable_scope("input") as scope:
		iterators = data.Dataset(size_index).get_train().repeat(EPOCH).batch(BATCH_SIZE).make_initializable_iterator()
		raw_features, labels = iterators.get_next()
		
		features = tf.reshape(
			tensor = raw_features,
			shape = [-1, data.FEATURE_HEIGHT[size_index], data.FEATURE_WIDTH[size_index], 1]
		)

	logits, softmax = new_cnn.infer(features, size_index, KEEP_PROB)

	with tf.variable_scope("train") as scope:
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

	summary_writer = tf.summary.FileWriter(TRAINING_LOG_DIR, tf.get_default_graph())

	saver = tf.train.Saver()

	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		iterators.initializer.run()

		tf.train.write_graph(sess.graph, CKPT_PATH, GRAPH_FILENAME)

		if tf.train.get_checkpoint_state(CKPT_PATH):
			saver.restore(sess, tf.train.latest_checkpoint(CKPT_PATH))

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
					save_path = CKPT_PATH + CKPT_PREFIX,
					global_step = _global_step
				)
				print("Model Saved")
		
		print("Training Done")

	summary_writer.close()

def main(unused_argv):
	train(size_index = CU_DEPTH)

if __name__ == "__main__":
	tf.app.run()
 