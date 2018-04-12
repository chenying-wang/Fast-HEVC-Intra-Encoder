import tensorflow as tf
import numpy as np

import data
import cnn

TRAINING_LOG_DIR = "./.tmp/train"
EPOCH = 1
SAMPLE_SIZE = 51520000
BATCH_SIZE = 64

CKPT_PATH = "./.tmp/"
CKPT_PREFIX = "training.ckpt"
CKPT_STEP = 1000

class Train:

	def train(self):

		global_step = tf.Variable(initial_value = 1, trainable = False, name = "global_step")

		iterators = []
		features = []
		labels = []
		size_index = []
		with tf.variable_scope("input") as scope:
			for i in range(len(data.FEATURE_WIDTH)):
				iterators.append(
					data.Dataset(i).get_train().repeat(EPOCH).batch(BATCH_SIZE).make_initializable_iterator()
				)
				raw_features, raw_labels = iterators[i].get_next()
				
				features.append(tf.reshape(
					tensor = raw_features,
					shape = [-1, data.FEATURE_HEIGHT[i], data.FEATURE_WIDTH[i], 1]
				))
				labels.append(raw_labels)
				size_index.append(tf.convert_to_tensor(i * np.ones([BATCH_SIZE]), dtype = tf.int32))

		with tf.variable_scope("input_control") as scope:
			rand = tf.random_uniform(
				shape = [],
				# minval = 0,
				# maxval = 11,
				minval = 3,
				maxval = 11,
				dtype = tf.int32
			)
			index = tf.case(
				[
					(tf.less(rand, 1), lambda: tf.constant(0)),
					(tf.greater(rand, 2), lambda: tf.constant(2))
				],
				default = lambda: tf.constant(1),
				name = "index"
			)
			fed_features, fed_labels, fed_size_index = tf.case(
				[
					(tf.less(rand, 1), lambda: (features[0], labels[0], size_index[0])),
					(tf.greater(rand, 2), lambda: (features[2], labels[2], size_index[2]))
				],
				default = lambda: (features[1], labels[1], size_index[1]),
				name = "fed_data"
			)

		logits, softmax = cnn.inference(fed_features, fed_size_index)

		with tf.variable_scope("train") as scope:
			pred = tf.argmax(logits, 1)

			onehot_lables = tf.one_hot(indices = fed_labels, depth = 2, dtype = tf.int64)
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
				labels = onehot_lables,
				logits = logits
			))
			
			accuracy = tf.reduce_mean(tf.cast(
				tf.equal(pred, fed_labels)
			, tf.float32))

			optimizer = tf.train.AdamOptimizer(learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999)
			train_batch = optimizer.minimize(loss, global_step = global_step)
		
		tf.summary.scalar("loss", loss, family = "train")
		tf.summary.scalar("accuracy", accuracy, family = "train")
		merged = tf.summary.merge_all()

		summary_writer = tf.summary.FileWriter(TRAINING_LOG_DIR, tf.get_default_graph())

		saver = tf.train.Saver()

		with tf.Session() as sess:
			tf.global_variables_initializer().run()
			for i in range(len(data.FEATURE_WIDTH)):
				iterators[i].initializer.run()

			if tf.train.get_checkpoint_state(CKPT_PATH):
				saver.restore(sess, tf.train.latest_checkpoint(CKPT_PATH))

			for _ in range (int(EPOCH * SAMPLE_SIZE / BATCH_SIZE)):
				_global_step = sess.run(global_step)

				_, _loss, _accuracy,  = sess.run([train_batch, loss, accuracy])
				print("Step %d, loss = %f, accuracy = %.2f%%, width = height = %d" % (_global_step, _loss, 100 * _accuracy, data.FEATURE_WIDTH[sess.run(index)]))
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
	Train().train()

if __name__ == "__main__":
	tf.app.run()
 