import sys
import tensorflow as tf

import data
import new_cnn

EVAL_LOG_DIR = "./.tmp/eval"
CU_SIZE = [64 * 64, 32 * 32, 16 * 16]
SAMPLE_SIZE = lambda size_index: int(1920 * 1024 / CU_SIZE[size_index]) * 500
BATCH_SIZE = 100

CKPT_PATH = lambda size_index: "./.tmp/" + str(size_index) + "/"

def eval(size_index):
	with tf.variable_scope("input") as scope:
		iterator = data.Dataset(size_index).get_eval().batch(BATCH_SIZE).make_initializable_iterator()
		raw_features, labels = iterator.get_next()
		features = tf.reshape(
			tensor = raw_features,
			shape = [-1, data.FEATURE_HEIGHT[size_index], data.FEATURE_WIDTH[size_index], 1]
		)

	logits, softmax = new_cnn.infer(features, size_index)
	pred = tf.argmax(logits, 1)
	
	accuracy = tf.reduce_mean(tf.cast(
		tf.equal(pred, labels)
	, tf.float32))
	
	saver = tf.train.Saver()

	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		iterator.initializer.run()
		
		if tf.train.get_checkpoint_state(CKPT_PATH(size_index)):
			saver.restore(sess, tf.train.latest_checkpoint(CKPT_PATH(size_index)))
		else:
			print("ERROR: CHECKPOINT NOT FOUND!")
			return

		confusion_sum = [[0, 0], [0, 0]]
		for idx in range(int(SAMPLE_SIZE(size_index) / BATCH_SIZE)):
			_pred, _labels, _accuracy = sess.run(
				[pred, labels, accuracy]
			)
				
			confusion = [[0, 0], [0, 0]]
			for i in range(BATCH_SIZE):
				l = _labels[i]
				p = _pred[i]
				confusion[l][p] += 1

			# print("Sample %d, accuracy = %.2f%%, unsplit_recall = %.2f%%, split_recall = %.2f%%" %
			# 	(
			# 		idx,
			# 		(100 * (confusion[0][0] + confusion[1][1]) / (sum(confusion[0]) + sum(confusion[1]))),
			# 		(100 * confusion[0][0] / sum(confusion[0])),
			# 		(100 * confusion[1][1] / sum(confusion[1]))
			# 	)
			# )
			# print(confusion)

			for i in range(2):
				for j in range(2):
					confusion_sum[i][j] += confusion[i][j]
			# print("total_accuracy = %.2f%%, total_unsplit_recall = %.2f%%, total_split_recall = %.2f%%" %
			# 	(
			# 		(100 * (confusion_sum[0][0] + confusion_sum[1][1]) / (sum(confusion_sum[0]) + sum(confusion_sum[1]))),
			# 		(100 * confusion_sum[0][0] / sum(confusion_sum[0])),
			#		(100 * confusion_sum[1][1] / sum(confusion_sum[1]))
			#	)
			# )
			print(confusion_sum)

		print("Evaluation Done")

def main(unused_argv):
	eval(size_index = int(sys.argv[1]))

if __name__ == "__main__":
	tf.app.run()
