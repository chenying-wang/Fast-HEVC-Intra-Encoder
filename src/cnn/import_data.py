import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

filenames = [""]

def _parse_line(line):
	RECORD_DEFAULT = [[0.0] for i in range(FEATURE_SIZE + LABEL_SIZE)]
	record = tf.convert_to_tensor(tf.decode_csv(line, RECORD_DEFAULT))
	features = tf.reshape(record[0 : FEATURE_SIZE], [FEATURE_WIDTH, FEATURE_HEIGHT])
	label = record[FEATURE_SIZE : FEATURE_SIZE + LABEL_SIZE]
	return features, label

FEATURE_WIDTH = 16
FEATURE_HEIGHT = 16
FEATURE_SIZE = FEATURE_WIDTH * FEATURE_HEIGHT
LABEL_SIZE = 1

dataset = tf.data.TextLineDataset(filenames).map(_parse_line)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()  

with tf.Session() as sess:
	sess.run(iterator.initializer)
	try:
		while True:
			print(sess.run(next_element))
	except tf.errors.OutOfRangeError:
		print("END!")
