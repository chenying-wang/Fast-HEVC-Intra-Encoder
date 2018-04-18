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

		confusion_sum = [[0, 0], [0, 0]]
		for idx in range(int(SAMPLE_SIZE / BATCH_SIZE)):
			_pred, _labels, _accuracy = sess.run(
				[pred, labels, accuracy]
			)
				
			confusion = [[0, 0], [0, 0]]
			for i in range(BATCH_SIZE):
				l = _labels[i]
				p = _pred[i]
				confusion[l][p] += 1

			print("Sample %d, accuracy = %.2f%%, unsplit_recall = %.2f%%, split_recall = %.2f%%" %
				(
					idx,
					(100 * (confusion[0][0] + confusion[1][1]) / (sum(confusion[0]) + sum(confusion[1]))),
					(100 * confusion[0][0] / sum(confusion[0])),
					(100 * confusion[1][1] / sum(confusion[1]))
				)
			)
			print(confusion)

			for i in range(2):
				for j in range(2):
					confusion_sum[i][j] += confusion[i][j]
			print("total_accuracy = %.2f%%, total_unsplit_recall = %.2f%%, total_split_recall = %.2f%%" %
				(
					(100 * (confusion_sum[0][0] + confusion_sum[1][1]) / (sum(confusion_sum[0]) + sum(confusion_sum[1]))),
					(100 * confusion_sum[0][0] / sum(confusion_sum[0])),
					(100 * confusion_sum[1][1] / sum(confusion_sum[1]))
				)
			)
			print(confusion_sum)

		print("Evaluation Done")

def main(unused_argv):
	eval(2)

if __name__ == "__main__":
	tf.app.run()
