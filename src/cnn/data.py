import tensorflow as tf

TRAINING_DATA_BASE_DIR = "../../training_dataset/"
DATA_NAME = "TrainingDataset"
FILENAME_SUFFIX = ".csv.gz"

EVAL_DATA_BASE_DIR = "../../test_dataset/"
EVAL_DATA_NAME = [
	"BasketballDrive_1920x1080_50_crop_64.csv.gz",
	"BasketballDrive_1920x1080_50_crop_32.csv.gz",
	"BasketballDrive_1920x1080_50_crop_16.csv.gz",
]

FRAME_WIDTH = [1024, 4096]
FRAME_HEIGHT = [576, 2176]
FRAME_RATE = 30
FRAME_LENGTH = 1000

FEATURE_WIDTH = [64, 32, 16]
FEATURE_HEIGHT = FEATURE_WIDTH

SHUFFLE = True

class Dataset:

	def __init__(self, size_index):
		self.size_index = size_index
		self.width = FEATURE_WIDTH[size_index]
		self.height = FEATURE_HEIGHT[size_index]
		self.size = self.width * self.height
		self.shuffle_buffer = 500

	def _parse_line(self, line):
		RECORD_DEFAULT = [[0.0] for i in range(self.size + 1)]
		record = tf.decode_csv(line, RECORD_DEFAULT)
		record_features = record[0 : self.size]
		record_labels = record[self.size]

		features = tf.reshape(record_features, [self.width, self.height])
		labels = tf.cast(record_labels, tf.int64)
		return features, labels

	def get_train(self):
		data_filenames = []
		
		for i in range(len(FRAME_WIDTH)):
			path = TRAINING_DATA_BASE_DIR
			path += DATA_NAME + "_" + str(FRAME_WIDTH[i]) + "x" + str(FRAME_HEIGHT[i]) + "_" 
			path += str(FRAME_RATE) + "_"
			path += str(self.width) + FILENAME_SUFFIX
			data_filenames.append(path)
		
		dataset = tf.data.TextLineDataset(data_filenames, compression_type = "GZIP")

		if SHUFFLE:
			return dataset.map(self._parse_line).shuffle(self.shuffle_buffer)
		return dataset.map(self._parse_line)

	def get_eval(self):
		data_filenames = [EVAL_DATA_BASE_DIR + EVAL_DATA_NAME[self.size_index]]
		dataset = tf.data.TextLineDataset(data_filenames, compression_type = "GZIP")
		return dataset.map(self._parse_line)
