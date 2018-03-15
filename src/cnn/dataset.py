import tensorflow as tf

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
LABEL_SIZE = 1

SHUFFLE_BUFFER = 10

RECORD_DEFAULT = [[0.0] for i in range(FEATURE_SIZE + LABEL_SIZE)]
def _parse_line(line):
	record = tf.convert_to_tensor(tf.decode_csv(line, RECORD_DEFAULT))
	features = tf.reshape(record[0 : FEATURE_SIZE], [FEATURE_WIDTH, FEATURE_HEIGHT])
	labels = tf.cast(record[FEATURE_SIZE : FEATURE_SIZE + LABEL_SIZE], tf.int32)
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
	return dataset.map(_parse_line)#.shuffle(SHUFFLE_BUFFER)
