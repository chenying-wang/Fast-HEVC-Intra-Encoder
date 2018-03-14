import tensorflow as tf

DATA_BASE_DIR = "..\\..\\training_dataset\\"
DATA_NAME = "TrainingDataset"
# RESOLUTION = ["1024x576", "2560x1440", "4096X2176"]
RESOLUTION = ["1024x576"]
FRAME_RATE = 30
FILENAME_SUFFIX = ".csv.gz"

FEATURE_WIDTH = 16
FEATURE_HEIGHT = FEATURE_WIDTH
FEATURE_SIZE = FEATURE_WIDTH * FEATURE_HEIGHT
LABEL_SIZE = 1

RECORD_DEFAULT = [[0.0] for i in range(FEATURE_SIZE + LABEL_SIZE)]
def _parse_line(line):
	record = tf.convert_to_tensor(tf.decode_csv(line, RECORD_DEFAULT))
	features = tf.reshape(record[0 : FEATURE_SIZE], [FEATURE_WIDTH, FEATURE_HEIGHT])
	labels = record[FEATURE_SIZE : FEATURE_SIZE + LABEL_SIZE]
	return {"features": features, "labels": labels}

def get_dataset():
	data_filenames = []
	for i in range(len(RESOLUTION)):
		path = DATA_BASE_DIR
		path += DATA_NAME + "_" + RESOLUTION[i] + "_" + str(FRAME_RATE) + "_"
		path += str(FEATURE_WIDTH) + FILENAME_SUFFIX
		data_filenames.append(path)
	dataset = tf.data.TextLineDataset(data_filenames, compression_type="GZIP")
	return dataset.map(_parse_line)
