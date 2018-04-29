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

	# Layer #3: 16x16x32 inception x2 16x16x128
	with tf.variable_scope("inception3") as scope:
		inception3a = _inception_16x16(
			input = conv1,
			input_channel = 32,
			branch_channel = [16, 32, 8, 8],
			reduced_channel = [16, 4],
			name = "inception3a"
		)
		inception3b = _inception_16x16(
			input = inception3a,
			input_channel = 64,
			branch_channel = [32, 64, 16, 16],
			reduced_channel = [32, 8],
			name = "inception3b"
		)

	# Pool #3: 16x16x128 max 3x3/2 8x8x128
	pool3 = _max_pooling(
		input = inception3b,
		pool_size = 3,
		strides = 2,
		name = "pool3"
	)

	# Layer #4: 8x8x128 inception x3 8x8x256
	with tf.variable_scope("inception4") as scope:
		inception4a = _inception_8x8(
			input = pool3,
			input_channel = 128,
			branch_channel = [48, 96, 24, 24],
			reduced_channel = [32, 8],
			name = "inception4a"
		)
		inception4b = _inception_8x8( #256
			input = inception4a,
			input_channel = 192,
			branch_channel = [64, 128, 32, 32],
			reduced_channel = [64, 8],
			name = "inception4b"
		)

	# Pool #4: 8x8x96avg 8x8/1v 1x1x96
	pool4 = _avg_pooling(
		input = inception4b,
		pool_size = 8,
		strides = 1,
		padding = "VALID",
		name = "pool4"
	)

	# Layer #5: 1x1x256 fc 2
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


def infer(features, size_index, keep_prob = 1.0):
	# Preprocess Layer
	with tf.variable_scope("preprocess") as scope:
		input_layer = tf.scalar_mul(
			scalar = 1 / 128,
			x = features - 128
		)

	# if size_index == 0:

	# elif size_index == 1:

	# elif size_index == 2:
	logits = _infer_16x16(input_layer, keep_prob)

	# Layers #6: Softmax 2
	with tf.variable_scope("softmax") as scope:
		softmax = tf.nn.softmax(
			logits = logits,
			name = scope.name
		)

	return logits, softmax
