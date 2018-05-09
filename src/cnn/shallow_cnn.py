import tensorflow as tf
import util

_conv2d = util.conv2d
_max_pooling = util.max_pooling
_avg_pooling = util.avg_pooling
_fc_with_dropout = util.fc_with_dropout
_inception_64x64 = util.inception_64x64
_inception_32x32 = util.inception_32x32
_inception_16x16 = util.inception_16x16
_inception_8x8 = util.inception_8x8

def _infer_64x64(features, keep_prob):
	# Layer #1: 64x64x1 conv 3x3/2 64x64x16
	conv1 = _conv2d(
		input = features,
		input_shape = [64, 64, 1],
		filters = 16,
		filter_size = [3, 3],
		biases = True,
		activation = True,
		name = "conv1"
	)

	# Pool #1: 64x64x16 max 3x3/2 32x32x16
	pool1 = _max_pooling(
		input = conv1,
		pool_size = 3,
		strides = 2,
		name = "pool1"
	)

	# Layer #2: 32x32x16 conv 3x3/2 32x32x64
	conv2= _conv2d(
		input = pool1,
		input_shape = [16, 16, 1],
		filters = 32,
		filter_size = [3, 3],
		biases = True,
		activation = True,
		name = "conv2"
	)

	# Pool #2: 32x32x64 max 3x3/2 16x16x64
	

	# Layer #4: 16x16x128 inception x1 16x16x192	
	with tf.variable_scope("inception4") as scope:
		inception4a = _inception_16x16(
			input = pool3,
			input_channel = 128,
			branch_channel = [48, 96, 24, 24],
			reduced_channel = [32, 8],
			name = "a"
		)
	
	# Pool #4: 16x16x192 max 3x3/2 8x8x192
	pool4 = _max_pooling(
		input = inception4a,
		pool_size = 3,
		strides = 2,
		name = "pool4"
	)

	# Layer #5: 8x8x192 inception x1 8x8x256
	with tf.variable_scope("inception5") as scope:
		inception5a = _inception_8x8( #256
			input = pool4,
			input_channel = 192,
			branch_channel = [64, 128, 32, 32],
			reduced_channel = [64, 8],
			name = "a"
		)

	# Pool #5: 8x8x256 avg 8x8/1v 1x1x256
	pool5 = _avg_pooling(
		input = inception5a,
		pool_size = 8,
		strides = 1,
		padding = "VALID",
		name = "pool4"
	)

	# Layer #6: 1x1x256 fc 2
	with tf.variable_scope("fc") as scope:
		pool5_flat = tf.reshape(
			tensor = pool5,
			shape = [-1, 256],
			name = "pool5_flat"
		)
		logits = _fc_with_dropout(
			input = pool5_flat,
			input_size = 256,
			output_size = 2,
			keep_prob = keep_prob,
			name = "logits"
		)

	return logits


def _infer_32x32(features, keep_prob):
	# Layer #1: 32x32x1 conv 3x3/2 32x32x32
	conv1 = _conv2d(
		input = features,
		input_shape = [16, 16, 1],
		filters = 32,
		filter_size = [3, 3],
		biases = True,
		activation = True,
		name = "conv1"
	)

	# Layer #2: 32x32x32 inception x1 32x32x64
	with tf.variable_scope("inception2") as scope:
		inception2a = _inception_32x32(
			input = conv1,
			input_channel = 32,
			branch_channel = [16, 32, 8, 8],
			reduced_channel = [16, 4],
			name = "a"
		)

	# Pool #2: 32x32x64 max 3x3/2 16x16x64
	pool2 = _max_pooling(
		input = inception2a,
		pool_size = 3,
		strides = 2,
		name = "pool2"
	)

	# Layer #3: 16x16x64 inception x2 16x16x192
	with tf.variable_scope("inception3") as scope:	
		inception3a = _inception_16x16(
			input = pool2,
			input_channel = 64,
			branch_channel = [32, 64, 16, 16],
			reduced_channel = [32, 8],
			name = "a"
		)
		inception3b = _inception_16x16(
			input = inception3a,
			input_channel = 128,
			branch_channel = [48, 96, 24, 24],
			reduced_channel = [32, 8],
			name = "b"
		)

	# Pool #3: 16x16x192 max 3x3/2 8x8x192
	pool3 = _max_pooling(
		input = inception3b,
		pool_size = 3,
		strides = 2,
		name = "pool3"
	)

	# Layer #4: 8x8x192 inception x1 8x8x256
	with tf.variable_scope("inception4") as scope:
		inception4a = _inception_8x8(
			input = pool3,
			input_channel = 192,
			branch_channel = [64, 128, 32, 32],
			reduced_channel = [64, 8],
			name = "a"
		)
	
	# Pool #4: 8x8x256 avg 8x8/1v 1x1x256
	pool4 = _avg_pooling(
		input = inception4a,
		pool_size = 8,
		strides = 1,
		padding = "VALID",
		name = "pool4"
	)

	# Layer #4: 1x1x256 fc 2
	with tf.variable_scope("fc") as scope:
		pool4_flat = tf.reshape(
			tensor = pool4,
			shape = [-1, 256],
			name = "pool4_flat"
		)
		logits = _fc_with_dropout(
			input = pool4_flat,
			input_size = 256,
			output_size = 2,
			keep_prob = keep_prob,
			name = "logits"
		)

	return logits


def _infer_16x16(features, keep_prob):
	# Layer #1: 16x16x1 conv 3x3/2 16x16x32
	conv1 = _conv2d(
		input = features,
		input_shape = [16, 16, 1],
		filters = 32,
		filter_size = [3, 3],
		biases = True,
		activation = True,
		name = "conv1"
	)

	# Layer #2: 16x16x32 inception x2 16x16x128
	with tf.variable_scope("inception2") as scope:
		inception2a = _inception_16x16(
			input = conv1,
			input_channel = 32,
			branch_channel = [16, 32, 8, 8],
			reduced_channel = [16, 4],
			name = "a"
		)
		inception2b = _inception_16x16(
			input = inception2a,
			input_channel = 64,
			branch_channel = [32, 64, 16, 16],
			reduced_channel = [32, 8],
			name = "b"
		)

	# Pool #2: 16x16x128 max 3x3/2 8x8x128
	pool2 = _max_pooling(
		input = inception2b,
		pool_size = 3,
		strides = 2,
		name = "pool2"
	)

	# Layer #3: 8x8x128 inception x2 8x8x256
	with tf.variable_scope("inception3") as scope:
		inception3a = _inception_8x8(
			input = pool2,
			input_channel = 128,
			branch_channel = [48, 96, 24, 24],
			reduced_channel = [32, 8],
			name = "a"
		)
		inception3b = _inception_8x8(
			input = inception3a,
			input_channel = 192,
			branch_channel = [64, 128, 32, 32],
			reduced_channel = [64, 8],
			name = "b"
		)

	# Pool #3: 8x8x256 avg 8x8/1v 1x1x256
	pool3 = _avg_pooling(
		input = inception3b,
		pool_size = 8,
		strides = 1,
		padding = "VALID",
		name = "pool3"
	)

	# Layer #4: 1x1x256 fc 2
	with tf.variable_scope("fc") as scope:
		pool3_flat = tf.reshape(
			tensor = pool3,
			shape = [-1, 256],
			name = "pool3_flat"
		)
		logits = _fc_with_dropout(
			input = pool3_flat,
			input_size = 256,
			output_size = 2,
			keep_prob = keep_prob,
			name = "logits"
		)

	return logits


def infer(features, size_index, keep_prob = 1.0):
	# Preprocess Layer
	with tf.variable_scope("preprocess") as scope:
		input_layer = tf.scalar_mul(
			scalar = 1 / 128,
			x = features - 128
		)

	# Deep ConvNet
	if size_index == 0:
		logits = _infer_64x64(input_layer, keep_prob)
	elif size_index == 1:
		logits = _infer_32x32(input_layer, keep_prob)
	elif size_index == 2:
		logits = _infer_16x16(input_layer, keep_prob)

	# Softmax Activation
	with tf.variable_scope("softmax") as scope:
		softmax = tf.nn.softmax(
			logits = logits,
			name = scope.name
		)

	return logits, softmax
