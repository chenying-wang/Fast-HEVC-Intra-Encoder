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
	strides = [1, 1],
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
			initializer = tf.random_normal_initializer(
				mean = 0.0,
				stddev = 2 / input_size
			)
		)
		conv = tf.nn.conv2d(
			input = input,
			filter = kernel,
			strides = [1, strides[0], strides[1], 1],
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

def _max_pooling(input, pool_size, strides, padding = "SAME", name = None):
	with tf.variable_scope(name) as scope:
		output = tf.nn.max_pool(
			input,
			ksize = [1, pool_size, pool_size, 1],
			strides = [1, strides, strides, 1],
			padding = padding,
			name = name
		)

	return output

def inference(features, size_index, keep_prob = 1.0):
	# Preprocess Layer
	with tf.variable_scope("preprocess") as scope:
		raw_input_layer_norm = tf.scalar_mul(
			scalar = 1 / 128,
			x = features - 128
		)

		input_layer_0 = tf.image.resize_images(
			images = raw_input_layer_norm,
			size = [64, 64],
			method = tf.image.ResizeMethod.BICUBIC
		)
		input_layer_1 = tf.image.resize_images(
			images = raw_input_layer_norm,
			size = [32, 32],
			method = tf.image.ResizeMethod.BICUBIC
		)
		input_layer_2 = tf.image.resize_images(
			images = raw_input_layer_norm,
			size = [16, 16],
			method = tf.image.ResizeMethod.BICUBIC
		)

		size_onehot = tf.one_hot(
			indices = size_index,
			depth = 3,
			name = "size_onehot"
		)

	# Layer #1-0: Conv 32x32x24
	with tf.variable_scope("layer_1_0") as scope:
		conv1_0_0 = _conv2d(
			input = input_layer_0,
			input_shape = [64, 64, 1],
			filters = 8,
			filter_size = [1, 1],
			biases = True,
			activation = True,
			name = "conv1_0_0"
		)
		conv1_0_1 = _conv2d(
			input = _conv2d(
				input = input_layer_0,
				input_shape = [64, 64, 1],
				filters = 4,
				filter_size = [1, 1],
				biases = False,
				name = "conv1_0_1_0"
			),
			input_shape = [64, 64, 4],
			filters = 8,
			filter_size = [3, 3],
			biases = True,
			activation = True,
			name = "conv1_0_1"
		)
		conv1_0_2 = _conv2d(
			input = _conv2d(
				input = _conv2d(
					input = input_layer_0,
					input_shape = [64, 64, 1],
					filters = 4,
					filter_size = [1, 1],
					biases = False,
					name = "conv1_0_2_0"
				),
				input_shape = [64, 64, 4],
				filters = 8,
				filter_size = [3, 3],
				biases = False,
				name = "conv1_0_2_1"
			),
			input_shape = [64, 64, 8],
			filters = 8,
			filter_size = [3, 3],
			biases = True,
			activation = True,
			name = "conv1_0_2"
		)
		conv1_0 = tf.concat(
			values = [conv1_0_0, conv1_0_1, conv1_0_2],
			axis = -1,
			name = "conv1_0"
		)
		pool1_0 = _max_pooling(
			input = conv1_0,
			pool_size = 2,
			strides = 2,
			name = "pool1_0"
		)

	# Layer #1-1: Conv 16x16x96
	with tf.variable_scope("layer1_1") as scope:
		conv1_1_0 = _conv2d(
			input = input_layer_1,
			input_shape = [32, 32, 1],
			filters = 32,
			filter_size = [1, 1],
			biases = True,
			activation = True,
			name = "conv1_1_0"
		)
		conv1_1_1 = _conv2d(
			input = _conv2d(
				input = input_layer_1,
				input_shape = [32, 32, 1],
				filters = 16,
				filter_size = [1, 1],
				biases = False,
				name = "conv1_1_1_0"
			),
			input_shape = [32, 32, 16],
			filters = 32,
			filter_size = [3, 3],
			biases = True,
			activation = True,
			name = "conv1_1_1"
		)
		conv1_1_2 = _conv2d(
			input = _conv2d(
				input = _conv2d(
					input = input_layer_1,
					input_shape = [32, 32, 1],
					filters = 16,
					filter_size = [1, 1],
					biases = False,
					name = "conv1_1_2_0"
				),
				input_shape = [32, 32, 16],
				filters = 32,
				filter_size = [3, 3],
				biases = False,
				name = "conv1_1_2_1"
			),
			input_shape = [32, 32, 32],
			filters = 32,
			filter_size = [3, 3],
			biases = True,
			activation = True,
			name = "conv1_1_2"
		)
		conv1_1 = tf.concat(
			values = [conv1_1_0, conv1_1_1, conv1_1_2],
			axis = -1,
			name = "conv1_1"
		)
		pool1_1 = _max_pooling(
			input = conv1_1,
			pool_size = 2,
			strides = 2,
			name = "pool1_1"
		)

	# Layer #1-2: Conv 8x8x384
	with tf.variable_scope("layer1_2") as scope:
		conv1_2_0 = _conv2d(
			input = input_layer_2,
			input_shape = [16, 16, 1],
			filters = 128,
			filter_size = [1, 1],
			biases = True,
			activation = True,
			name = "conv1_2_0"
		)
		conv1_2_1 = _conv2d(
			input = _conv2d(
				input = input_layer_2,
				input_shape = [16, 16, 1],
				filters = 64,
				filter_size = [1, 1],
				biases = False,
				name = "conv1_2_1_0"
			),
			input_shape = [16, 16, 64],
			filters = 128,
			filter_size = [3, 3],
			biases = True,
			activation = True,
			name = "conv1_2_1"
		)
		conv1_2_2 = _conv2d(
			input = _conv2d(
				input = _conv2d(
					input = input_layer_2,
					input_shape = [16, 16, 1],
					filters = 64,
					filter_size = [1, 1],
					biases = False,
					name = "conv1_2_2_0"
				),
				input_shape = [16, 16, 64],
				filters = 128,
				filter_size = [3, 3],
				biases = False,
				name = "conv1_2_2_1"
			),
			input_shape = [16, 16, 128],
			filters = 128,
			filter_size = [3, 3],
			biases = True,
			activation = True,
			name = "conv1_2_2"
		)
		conv1_2 = tf.concat(
			values = [conv1_2_0, conv1_2_1, conv1_2_2],
			axis = -1,
			name = "conv1_2"
		)
		pool1_2 = _max_pooling(
			input = conv1_2,
			pool_size = 2,
			strides = 2,
			name = "pool1_2"
		)

	# Layer #2-0: Conv 16x16x32
	with tf.variable_scope("layer2_0"):
		conv2_0 = _conv2d(
			input = pool1_0,
			input_shape = [32, 32, 24],
			filters = 32,
			filter_size = [3, 3],
			activation = True,
			name = "conv2_0"
		)
		pool2_0 = _max_pooling(
			input = conv2_0,
			pool_size = 2,
			strides = 2,
			name = "pool2_0"
		)

	# Layer #2-1: Conv 8x8x128
	with tf.variable_scope("layer2_1"):
		conv2_1 = _conv2d(
			input = pool1_1,
			input_shape = [16, 16, 96],
			filters = 128,
			filter_size = [3, 3],
			activation = True,
			name = "conv2_1"
		)
		pool2_1 = _max_pooling(
			input = conv2_1,
			pool_size = 2,
			strides = 2,
			name = "pool2_1"
		)

	# Layer #2-2: Conv 4x4x512
	with tf.variable_scope("layer2_2"):
		conv2_2 = _conv2d(
			input = pool1_2,
			input_shape = [8, 8, 384],
			filters = 512,
			filter_size = [3, 3],
			activation = True,
			name = "conv2_2"
		)
		pool2_2 = _max_pooling(
			input = conv2_2,
			pool_size = 2,
			strides = 2,
			name = "pool2_2"
		)
	
	# Layer #3-0: Conv 8x8x16
	with tf.variable_scope("layer3_0"):
		conv3_0 = _conv2d(
			input = pool2_0,
			input_shape = [16, 16, 32],
			filters = 16,
			filter_size = [1, 1],
			activation = True,
			name = "conv3_0"
		)
		pool3_0 = _max_pooling(
			input = conv3_0,
			pool_size = 2,
			strides = 2,
			name = "pool3_0"
		)
	
	# Layer #3-1: Conv 4x4x64
	with tf.variable_scope("layer3_1"):
		conv3_1 = _conv2d(
			input = pool2_1,
			input_shape = [8, 8, 128],
			filters = 64,
			filter_size = [1, 1],
			activation = True,
			name = "conv3_1"
		)
		pool3_1 = _max_pooling(
			input = conv3_1,
			pool_size = 2,
			strides = 2,
			name = "pool3_1"
		)

	# Layer #3-2: Conv 2x2x256
	with tf.variable_scope("layer3_2"):
		conv3_2 = _conv2d(
			input = pool2_2,
			input_shape = [4, 4, 512],
			filters = 256,
			filter_size = [1, 1],
			activation = True,
			name = "conv3_2"
		)
		pool3_2 = _max_pooling(
			input = conv3_2,
			pool_size = 2,
			strides = 2,
			name = "pool3_2"
		)

	# Layer #4: Concat 3072 + 3
	with tf.variable_scope("concat") as scope:
		pool3_0_flat = tf.reshape(
			tensor = pool3_0,
			shape = [-1, 1024],
			name = "pool3_0_flat"
		)
		pool3_1_flat = tf.reshape(
			tensor = pool3_1,
			shape = [-1, 1024],
			name = "pool3_1_flat"
		)
		pool3_2_flat = tf.reshape(
			tensor = pool3_2,
			shape = [-1, 1024],
			name = "pool3_2_flat"
		)
		concat = tf.concat(
			values = [pool3_0_flat, pool3_1_flat, pool3_2_flat, size_onehot],
			axis = 1,
			name = scope.name
		)

	'''
	# Layer #5: Dense 256 + 3
	with tf.variable_scope("dense1") as scope:
		weights = tf.get_variable(
			name = "weights",
			shape = [3075, 256],
			dtype = tf.float32,
			initializer = tf.random_normal_initializer(mean = 0.0, stddev = 2 / 3072)
		)
		biases = tf.get_variable(
			name = 'biases',
			shape = [256],
			initializer = tf.zeros_initializer()
		)
		activation = tf.nn.leaky_relu(
			features = tf.matmul(concat, weights) + biases,
			alpha = 0.01
		)
		dense1 = tf.concat(
			values = [activation, size_onehot],
			axis = 1,
			name = scope.name
		)

	# Layer #6: Dense 24 + 3
	with tf.variable_scope("dense2") as scope:
		weights = tf.get_variable(
			name = "weights",
			shape = [259, 24],
			dtype = tf.float32,
			initializer = tf.random_normal_initializer(mean = 0.0, stddev = 2 / 256)
		)
		biases = tf.get_variable(
			name = 'biases',
			shape = [24],
			initializer = tf.zeros_initializer()
		)
		activation = tf.nn.leaky_relu(
			features = tf.matmul(dense1, weights) + biases,
			alpha = 0.01
		)
		dense2 = tf.concat(
			values = [activation, size_onehot],
			axis = 1,
			name = scope.name
		)
	'''

	concat_dropout = tf.nn.dropout(
		x = concat,
		keep_prob = keep_prob,
		name = "dropout"
	)

	# Layers #7: Logits 2
	with tf.variable_scope("logits") as scope:
		weights = tf.get_variable(
			name = "weights",
			shape = [3075, 2],
			dtype = tf.float32,
			initializer = tf.random_normal_initializer(mean = 0.0, stddev = 2 / 3075)
		)
		biases = tf.get_variable(
			name = 'biases',
			shape = [2],
			initializer = tf.zeros_initializer()
		)
		
		logits = _activate(
			features = tf.matmul(concat_dropout, weights) + biases,
			name = scope.name
		)

	# Layers #8: Softmax 2
	with tf.variable_scope("softmax") as scope:
		softmax = tf.nn.softmax(
			logits = logits,
			name = scope.name
		)

	return logits, softmax
