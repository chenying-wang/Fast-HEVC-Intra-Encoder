import tensorflow as tf
import numpy as np

DATA_BASE_DIR = "..\\..\\training_dataset\\"
DATA_NAME = "TrainingDataset"
FILENAME_SUFFIX = ".csv.gz"

FRAME_WIDTH = [1024, 2560, 4096]
FRAME_HEIGHT = [576, 1440, 2176]
FRAME_WIDTH = [1024]
FRAME_HEIGHT = [576]
FRAME_RATE = 30
FRAME_LENGTH = 1000

FEATURE_WIDTH = 16
FEATURE_HEIGHT = FEATURE_WIDTH
FEATURE_SIZE = FEATURE_WIDTH * FEATURE_HEIGHT

SHUFFLE_BUFFER = int(8 * max(FRAME_WIDTH) / FEATURE_WIDTH * max(FRAME_HEIGHT) / FEATURE_HEIGHT)

RECORD_DEFAULT = [[0.0] for i in range(FEATURE_SIZE + 1)]
def _parse_line(line):
	record = tf.decode_csv(line, RECORD_DEFAULT)
	record_features = record[0 : FEATURE_SIZE]
	record_features_norm = record_features - tf.reduce_mean(record_features)
	record_features_norm = tf.scalar_mul(1/128, record_features_norm)
	record_labels = record[FEATURE_SIZE]

	features = tf.reshape(record_features_norm, [FEATURE_WIDTH, FEATURE_HEIGHT])
	labels = tf.cast(record_labels, tf.int64)
	return features, labels

def get():
	data_filenames = []
	for i in range(len(FRAME_WIDTH)):
		path = DATA_BASE_DIR
		path += DATA_NAME + "_" + str(FRAME_WIDTH[i]) + "x" + str(FRAME_HEIGHT[i]) + "_" 
		path += str(FRAME_RATE) + "_"
		path += str(FEATURE_WIDTH) + FILENAME_SUFFIX
		data_filenames.append(path)
	dataset = tf.data.TextLineDataset(data_filenames, compression_type="GZIP")
	return dataset.map(_parse_line).shuffle(SHUFFLE_BUFFER)
