import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_BASE_DIR = "..\\..\\training_dataset\\"
DATA_NAME = "TrainingDataset"
RESOLUTION = ["1024x576", "2560x1440", "4096X2176"]
FRAME_RATE = 30
FILENAME_EXTENSION = ".csv"
SAMPLE_LENGTH = 1

FEATURE_WIDTH = 64
FEATURE_HEIGHT = FEATURE_WIDTH
FEATURE_SIZE = FEATURE_WIDTH * FEATURE_HEIGHT
LABEL_SIZE = 1

def _parse_line(line):
	RECORD_DEFAULT = [[0.0] for i in range(FEATURE_SIZE + LABEL_SIZE)]
	record = tf.convert_to_tensor(tf.decode_csv(line, RECORD_DEFAULT))
	features = tf.reshape(record[0 : FEATURE_SIZE], [FEATURE_WIDTH, FEATURE_HEIGHT])
	label = record[FEATURE_SIZE : FEATURE_SIZE + LABEL_SIZE]
	return features, label

data_filenames = []
for i in range(len(RESOLUTION)):
	path = DATA_BASE_DIR
	path += DATA_NAME + "_" + RESOLUTION[i] + "_" + str(FRAME_RATE) + "_"
	path += str(FEATURE_WIDTH) + FILENAME_EXTENSION
	data_filenames.append(path)

dataset = tf.data.TextLineDataset(data_filenames).map(_parse_line)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()  

with tf.Session() as sess:
	sess.run(iterator.initializer)
	try:
		for i in range (SAMPLE_LENGTH):
			print(sess.run(next_element))
	except tf.errors.OutOfRangeError:
		print("END!")
