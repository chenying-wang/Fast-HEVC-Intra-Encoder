import tensorflow as tf
import numpy as np

import data
import cnn

EVAL_LOG_DIR = "./.tmp/eval"
SAMPLE_SIZE = 81000
BATCH_SIZE = 1000

CKPT_PATH = "./.tmp/"

def eval(depth):
	with tf.variable_scope("input") as scope:
		iterator = data.Dataset(depth).get_eval().batch(BATCH_SIZE).make_initializable_iterator()
		raw_features, labels = iterator.get_next()
		features = tf.reshape(
			tensor = raw_features,
			shape = [-1, data.FEATURE_HEIGHT[depth], data.FEATURE_WIDTH[depth], 1]
		)

	size_index = tf.convert_to_tensor(depth * np.ones([BATCH_SIZE]), dtype = tf.int32)
	logits, softmax = cnn.inference(features, size_index)
	pred = tf.argmax(logits, 1)
	
	accuracy = tf.reduce_mean(tf.cast(
		tf.equal(pred, labels)
	, tf.float32))
	
	saver = tf.train.Saver()

	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		iterator.initializer.run()
		
		if tf.train.get_checkpoint_state(CKPT_PATH):
			saver.restore(sess, tf.train.latest_checkpoint(CKPT_PATH))
	
		correct_count = 0
		precision = 0.0
		for count in range (int(SAMPLE_SIZE / BATCH_SIZE)):
			acc = sess.run(accuracy)
			correct_count += acc * BATCH_SIZE
			precision = 100 * correct_count / ((count + 1) * BATCH_SIZE)
			print("Sample %d, total_accuracy = %.2f%%, correct = %.1f%%" %
				(count * BATCH_SIZE, precision, 100 * acc)
			)

		print("Evaluation Done")

def main(unused_argv):
	eval(2)

if __name__ == "__main__":
	tf.app.run()
