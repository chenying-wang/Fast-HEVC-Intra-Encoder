import tensorflow as tf

def _activate(features, name = None):
	return tf.nn.leaky_relu(
		features = features,
		alpha = 0.01,
		name = name
	)

def _conv2d(input,
			input_shape,
			filters,
			filter_size,
			strides = 1,
			biases = True,
			activation = False,
			name = None):

	with tf.variable_scope(name):
		input_size = input_shape[0] * input_shape[1] * input_shape[2]
		input_channels = input_shape[2]

		kernel = tf.get_variable(
			name = "weights",
			shape = [filter_size[0], filter_size[1], input_channels, filters],
			dtype = tf.float32,
			initializer = tf.glorot_uniform_initializer(
				dtype = tf.float32
			)
		)
		conv = tf.nn.conv2d(
			input = input,
			filter = kernel,
			strides = [1, strides, strides, 1],
			padding = "SAME",
			name = "conv"
		)

		if biases:
			biases = tf.get_variable(
				name = "biases",
				shape = filters,
				dtype = tf.float32,
				initializer = tf.zeros_initializer()
			)
			pre_activation = tf.nn.bias_add(
				value = conv,
				bias = biases,
				data_format = "NHWC"
			)
		else:
			pre_activation = conv

		if activation:
			output = _activate(pre_activation)
		else:
			output = pre_activation
	
	return output

def _max_pooling(input,
				 pool_size,
				 strides,
				 padding = "SAME",
				 name = None):
	with tf.variable_scope(name) as scope:
		output = tf.nn.max_pool(
			input,
			ksize = [1, pool_size, pool_size, 1],
			strides = [1, strides, strides, 1],
			padding = padding,
			name = name
		)

	return output

def _avg_pooling(input,
				 pool_size,
				 strides,
				 padding = "SAME",
				 name = None):
	with tf.variable_scope(name) as scope:
		output = tf.nn.avg_pool(
			input,
			ksize = [1, pool_size, pool_size, 1],
			strides = [1, strides, strides, 1],
			padding = padding,
			name = name
		)

	return output

def _fc_with_dropout(input,
					 input_size,
					 output_size,
					 keep_prob = 1.0,
					 name = None):
	with tf.variable_scope(name) as scope:
		weights = tf.get_variable(
			name = "weights",
			shape = [input_size, output_size],
			dtype = tf.float32,
			initializer = tf.random_normal_initializer(mean = 0.0, stddev = 2 / input_size)
		)
		biases = tf.get_variable(
			name = 'biases',
			shape = [output_size],
			initializer = tf.zeros_initializer()
		)
		input_dropout = tf.nn.dropout(
			x = input,
			keep_prob = keep_prob
		)
		output = _activate(
			features = tf.matmul(input_dropout, weights) + biases,
			name = scope.name
		)
	
	return output

def _inception(input,
			  width,
			  height,
			  input_channel,
			  branch_channel,
			  reduced_channel,
			  name = None):
	with tf.variable_scope(name) as scope:
		branch0 = _conv2d(
			input = input,
			input_shape = [width, height, input_channel],
			filters = branch_channel[0],
			filter_size = [1, 1],
			biases = True,
			activation = True,
			name = name + "_1x1"
		)
		branch1 = _conv2d(
			input = _conv2d(
				input = input,
				input_shape = [width, height, input_channel],
				filters = reduced_channel[1],
				filter_size = [1, 1],
				biases = True,
				activation = True,
				name = name + "_3x3_reduced"
			),
			input_shape = [width, height, reduced_channel[1]],
			filters = branch_channel[1],
			filter_size = [3, 3],
			biases = True,
			activation = True,
			name = name + "_3x3"
		)
		branch2 = _conv2d(
			input = _conv2d(
				input = _conv2d(
					input = input,
					input_shape = [width, height, input_channel],
					filters = reduced_channel[1],
					filter_size = [1, 1],
					biases = True,
					activation = True,
					name = name + "_5x5_reduced"
				),
				input_shape = [width, height, reduced_channel[1]],
				filters = branch_channel[2],
				filter_size = [3, 3],
				biases = True,
				activation = True,
				name = name + "_5x5_0"
			),
			input_shape = [width, height, branch_channel[2]],
			filters = branch_channel[2],
			filter_size = [3, 3],
			biases = True,
			activation = True,
			name = name + "_5x5"
		)
		branch3 = _conv2d(
			input = _max_pooling(
				input = input,
				pool_size = 3,
				strides = 1,
				name = name + "_pool"
			),
			input_shape = [width, height, input_channel],
			filters = branch_channel[3],
			filter_size = [1, 1],
			biases = True,
			activation = True,
			name = name + "_pool_reduced"
		)
		output = tf.concat(
			values = [branch0, branch1, branch2, branch3],
			axis = -1,
			name = name
		)

	return output

def _inception_64x64(input, input_channel, branch_channel, reduced_channel, name = None):
	return _inception(
		input = input,
		width = 64,
		height = 64,
		input_channel = input_channel,
		branch_channel = branch_channel,
		reduced_channel = reduced_channel,
		name = name
	)

def _inception_32x32(input, input_channel, branch_channel, reduced_channel, name = None):
	return _inception(
		input = input,
		width = 32,
		height = 32,
		input_channel = input_channel,
		branch_channel = branch_channel,
		reduced_channel = reduced_channel,
		name = name
	)

def _inception_16x16(input, input_channel, branch_channel, reduced_channel, name = None):
	return _inception(
		input = input,
		width = 16,
		height = 16,
		input_channel = input_channel,
		branch_channel = branch_channel,
		reduced_channel = reduced_channel,
		name = name
	)

def _inception_8x8(input, input_channel, branch_channel, reduced_channel, name = None):
	return _inception(
		input = input,
		width = 8,
		height = 8,
		input_channel = input_channel,
		branch_channel = branch_channel,
		reduced_channel = reduced_channel,
		name = name
	)

def inference(features, size_index, keep_prob = 1.0):
	# Preprocess Layer
	with tf.variable_scope("preprocess") as scope:
		raw_input_layer_norm = tf.scalar_mul(
			scalar = 1 / 128,
			x = features - 128
		)

		# input_layer = tf.image.resize_images(
		# 	images = raw_input_layer_norm,
		# 	size = [64, 64],
		# 	method = tf.image.ResizeMethod.BICUBIC
		# )
		input_layer = raw_input_layer_norm

		size_onehot = tf.one_hot(
			indices = size_index,
			depth = 3,
			name = "size_onehot"
		)

	# Layer #1: 16x16x1 conv 3x3/2 16x16x32
	conv1 = _conv2d(
		input = input_layer,
		input_shape = [16, 16, 1],
		filters = 32,
		filter_size = [3, 3],
		biases = True,
		activation = True,
		name = "conv1"
	)

	# Layer #1: 64x64x1 conv 5x5/2 64x64x12
	# conv1 = _conv2d(
	# 	input = input_layer,
	# 	input_shape = [64, 64, 1],
	# 	filters = 12,
	# 	filter_size = [5, 5],
	# 	biases = True,
	# 	activation = True,
	# 	name = "conv1"
	# )
	
	# Pool #1: 64x64x12 max 3x3/2 32x32x12
	# pool1 = _max_pooling(
	# 	input = conv1,
	# 	pool_size = 3,
	# 	strides = 2,
	# 	name = "pool1"
	# )

	# Layer #2: 32x32x12 conv 3x3/1 32x32x32
	# conv2 = _conv2d(
	# 	input = _conv2d(
	# 		input = pool1,
	# 		input_shape = [32, 32, 12],
	# 		filters = 12,
	# 		filter_size = [1, 1],
	# 		biases = True,
	# 		activation = True,
	# 		name = "conv2_reduced"
	# 	),
	# 	input_shape = [32, 32, 12],
	# 	filters = 32,
	# 	filter_size = [3, 3],
	# 	biases = True,
	# 	activation = True,
	# 	name = "conv2"
	# )

	# Pool #2: 32x32x32 max 3x3/2 16x16x32
	# pool2 = _max_pooling(
	# 	input = conv2,
	# 	pool_size = 3,
	# 	strides = 2,
	# 	name = "pool2"
	# )

	# Layer #3: 16x16x32 inception x2 16x16x90
	with tf.variable_scope("inception3") as scope:
		inception3a = _inception_16x16(
			input = conv1,
			input_channel = 32,
			branch_channel = [12, 24, 6, 6],
			reduced_channel = [18, 3],
			name = "inception3a"
		)
		inception3b = _inception_16x16(
			input = inception3a,
			input_channel = 48,
			branch_channel = [24, 36, 18, 12],
			reduced_channel = [24, 6],
			name = "inception3b"
		)

	# Pool #3: 16x16x90 max 3x3/2 8x8x90
	pool3 = _max_pooling(
		input = inception3b,
		pool_size = 3,
		strides = 2,
		name = "pool3"
	)

	# Layer #4: 8x8x90 inception x3 8x8x156
	with tf.variable_scope("inception4") as scope:
		inception4a = _inception_8x8(
			input = pool3,
			input_channel = 90,
			branch_channel = [36, 39, 9, 12],
			reduced_channel = [18, 3],
			name = "inception4a"
		)
		inception4b = _inception_8x8(
			input = inception4a,
			input_channel = 96,
			branch_channel = [24, 48, 12, 12],
			reduced_channel = [24, 4],
			name = "inception4b"
		)
		# inception4c = _inception_8x8(
		# 	input = inception4b,
		# 	input_channel = 96,
		# 	branch_channel = [48, 60, 24, 24],
		# 	reduced_channel = [30, 6],
		# 	name = "inception4c"
		# )

	# Pool #4: 8x8x96avg 8x8/1v 1x1x96
	pool4 = _avg_pooling(
		input = inception4b,
		pool_size = 8,
		strides = 1,
		padding = "VALID",
		name = "pool4"
	)

	# Layer #5: 1x1x156 fc 2
	with tf.variable_scope("fc") as scope:
		pool4_flat = tf.reshape(
			tensor = pool4,
			shape = [-1, 96],
			name = "pool4_flat"
		)
		logits = _fc_with_dropout(
			input = pool4_flat,
			input_size = 96,
			output_size = 2,
			keep_prob = keep_prob,
			name = "logits"
		)

	# Layers #6: Softmax 2
	with tf.variable_scope("softmax") as scope:
		softmax = tf.nn.softmax(
			logits = logits,
			name = scope.name
		)

	return logits, softmax
