import tensorflow as tf

def inference(features, width, height):
	# Preprocess Layer
	with tf.variable_scope("preprocess") as scope:
		raw_input_layer = tf.reshape(
			tensor = features,
			shape = [-1, height, width, 1]
		)
		raw_input_layer_norm = tf.scalar_mul(
			scalar = 1/128,
			x = raw_input_layer - tf.reduce_mean(features)
		)

		input_layer_0 = tf.image.resize_bicubic(
			images = raw_input_layer_norm,
			size = [64, 64]
		)
		input_layer_1 = tf.image.resize_bicubic(
			images = raw_input_layer_norm,
			size = [32, 32]
		)
		input_layer_2 = tf.image.resize_bicubic(
			images = raw_input_layer_norm,
			size = [16, 16]
		)

	# Layer #1-0: Conv 16x16x8
	with tf.variable_scope("conv1_0") as scope:
		kernel = tf.get_variable(
			name = "weights",
			shape = [4, 4, 1, 8],
			dtype = tf.float32,
			initializer = tf.random_normal_initializer(mean = 0.0, stddev = 2 / 4096)
		)
		conv = tf.nn.conv2d(
			input = input_layer_0,
			filter = kernel,
			strides = [1, 1, 1, 1],
			padding = "SAME"
		)
		biases = tf.get_variable(
			name = "biases",
			shape = [8],
			dtype = tf.float32,
			initializer = tf.zeros_initializer()
		)
		pre_activation = tf.nn.bias_add(
			value = conv,
			bias = biases,
			data_format = "NHWC"			
		)
		conv1_0 = tf.nn.relu(
			features = pre_activation,
			name = scope.name
		)
	with tf.variable_scope("pool1_2") as scope:
		pool1_0 = tf.nn.max_pool(
			value = conv1_0,
			ksize = [1, 4, 4, 1],
			strides = [1, 4, 4, 1],
			padding = "SAME",
			name = scope.name
		)

	# Layer #1-1: Conv 8x8x32
	with tf.variable_scope("conv1_1") as scope:
		kernel = tf.get_variable(
			name = "weights",
			shape = [4, 4, 1, 32],
			dtype = tf.float32,
			initializer = tf.random_normal_initializer(mean = 0.0, stddev = 2 / 1024)
		)
		conv = tf.nn.conv2d(
			input = input_layer_1,
			filter = kernel,
			strides = [1, 1, 1, 1],
			padding = "SAME"
		)
		biases = tf.get_variable(
			name = "biases",
			shape = [32],
			dtype = tf.float32,
			initializer = tf.zeros_initializer()
		)
		pre_activation = tf.nn.bias_add(
			value = conv,
			bias = biases,
			data_format = "NHWC"			
		)
		conv1_1 = tf.nn.relu(
			features = pre_activation,
			name = scope.name
		)
	with tf.variable_scope("pool1_2") as scope:
		pool1_1 = tf.nn.max_pool(
			value = conv1_1,
			ksize = [1, 4, 4, 1],
			strides = [1, 4, 4, 1],
			padding = "SAME",
			name = scope.name
		)

	# Layer #1-2: Conv 4x4x128
	with tf.variable_scope("conv1_2") as scope:
		kernel = tf.get_variable(
			name = "weights",
			shape = [4, 4, 1, 128],
			dtype = tf.float32,
			initializer = tf.random_normal_initializer(mean = 0.0, stddev = 2 / 256)
		)
		conv = tf.nn.conv2d(
			input = input_layer_2,
			filter = kernel,
			strides = [1, 1, 1, 1],
			padding = "SAME"
		)
		biases = tf.get_variable(
			name = "biases",
			shape = [128],
			dtype = tf.float32,
			initializer = tf.zeros_initializer()
		)
		pre_activation = tf.nn.bias_add(
			value = conv,
			bias = biases,
			data_format = "NHWC"			
		)
		conv1_2 = tf.nn.relu(
			features = pre_activation,
			name = scope.name
		)
	with tf.variable_scope("pool1_2") as scope:
		pool1_2 = tf.nn.max_pool(
			value = conv1_2,
			ksize = [1, 4, 4, 1],
			strides = [1, 4, 4, 1],
			padding = "SAME",
			name = scope.name
		)

	# Layer #2-0: Conv 8x8x16
	with tf.variable_scope("conv2_0") as scope:
		kernel = tf.get_variable(
			name = "weights",
			shape = [2, 2, 8, 16],
			dtype = tf.float32,
			initializer = tf.random_normal_initializer(mean = 0.0, stddev = 2 / 2048)
		)
		conv = tf.nn.conv2d(
			input = pool1_0,
			filter = kernel,
			strides = [1, 1, 1, 1],
			padding = "SAME"
		)
		biases = tf.get_variable(
			name = "biases",
			shape = [16],
			dtype = tf.float32,
			initializer = tf.zeros_initializer()
		)
		pre_activation = tf.nn.bias_add(
			value = conv,
			bias = biases,
			data_format = "NHWC"
		)
		conv2_0 = tf.nn.relu(
			features = pre_activation,
			name = scope.name
		)
	with tf.variable_scope("pool2_0") as scope:
		pool2_0 = tf.nn.max_pool(
			value = conv2_0,
			ksize = [1, 2, 2, 1],
			strides = [1, 2, 2, 1],
			padding = "SAME",
			name = scope.name
		)

	# Layer #2-1: Conv 4x4x64
	with tf.variable_scope("conv2_1") as scope:
		kernel = tf.get_variable(
			name = "weights",
			shape = [2, 2, 32, 64],
			dtype = tf.float32,
			initializer = tf.random_normal_initializer(mean = 0.0, stddev = 2 / 2048)
		)
		conv = tf.nn.conv2d(
			input = pool1_1,
			filter = kernel,
			strides = [1, 1, 1, 1],
			padding = "SAME"
		)
		biases = tf.get_variable(
			name = "biases",
			shape = [64],
			dtype = tf.float32,
			initializer = tf.zeros_initializer()
		)
		pre_activation = tf.nn.bias_add(
			value = conv,
			bias = biases,
			data_format = "NHWC"
		)
		conv2_1 = tf.nn.relu(
			features = pre_activation,
			name = scope.name
		)
	with tf.variable_scope("pool2_1") as scope:
		pool2_1 = tf.nn.max_pool(
			value = conv2_1,
			ksize = [1, 2, 2, 1],
			strides = [1, 2, 2, 1],
			padding = "SAME",
			name = scope.name
		)
	
	# Layer #2-2: Conv 2x2x256
	with tf.variable_scope("conv2_2") as scope:
		kernel = tf.get_variable(
			name = "weights",
			shape = [2, 2, 128, 256],
			dtype = tf.float32,
			initializer = tf.random_normal_initializer(mean = 0.0, stddev = 2 / 2048)
		)
		conv = tf.nn.conv2d(
			input = pool1_2,
			filter = kernel,
			strides = [1, 1, 1, 1],
			padding = "SAME"
		)
		biases = tf.get_variable(
			name = "biases",
			shape = [256],
			dtype = tf.float32,
			initializer = tf.zeros_initializer()
		)
		pre_activation = tf.nn.bias_add(
			value = conv,
			bias = biases,
			data_format = "NHWC"
		)
		conv2_2 = tf.nn.relu(
			features = pre_activation,
			name = scope.name
		)
	with tf.variable_scope("pool2_2") as scope:
		pool2_2 = tf.nn.max_pool(
			value = conv2_2,
			ksize = [1, 2, 2, 1],
			strides = [1, 2, 2, 1],
			padding = "SAME",
			name = scope.name
		)

	# Layer #3: Concat 3072
	with tf.variable_scope("concat") as scope:
		pool2_0_flat = tf.reshape(
			tensor = pool2_0,
			shape = [-1, 1024]
		)
		pool2_1_flat = tf.reshape(
			tensor = pool2_1,
			shape = [-1, 1024]
		)
		pool2_2_flat = tf.reshape(
			tensor = pool2_2,
			shape = [-1, 1024]
		)
		concat = tf.concat(
			values = [pool2_0_flat, pool2_1_flat, pool2_2_flat],
			axis = 1,
			name = scope.name
		)
	
	# Layer #4: Dense 256
	with tf.variable_scope("dense1") as scope:
		weights = tf.get_variable(
			name = "weights",
			shape = [3072, 256],
			dtype = tf.float32,
			initializer = tf.random_normal_initializer(mean = 0.0, stddev = 2 / 3072)
		)
		biases = tf.get_variable(
			name = 'biases',
			shape = [256],
			initializer = tf.zeros_initializer()
		)
		dense1 = tf.nn.relu(
			features = tf.matmul(concat, weights) + biases,
			name = scope.name
		)

	# Layer #5: Dense 24
	with tf.variable_scope("dense2") as scope:
		weights = tf.get_variable(
			name = "weights",
			shape = [256, 24],
			dtype = tf.float32,
			initializer = tf.random_normal_initializer(mean = 0.0, stddev = 2 / 256)
		)
		biases = tf.get_variable(
			name = 'biases',
			shape = [24],
			initializer = tf.zeros_initializer()
		)
		dense2 = tf.nn.relu(
			features = tf.matmul(dense1, weights) + biases,
			name = scope.name
		)

	# Layers #6: Logits 2
	with tf.variable_scope("logits") as scope:
		weights = tf.get_variable(
			name = "weights",
			shape = [24, 2],
			dtype = tf.float32,
			initializer = tf.random_normal_initializer(mean = 0.0, stddev = 2 / 24)
		)
		biases = tf.get_variable(
			name = 'biases',
			shape = [2],
			initializer = tf.zeros_initializer()
		)
		logits = tf.nn.relu(
			features = tf.matmul(dense2, weights) + biases,
			name = scope.name
		)

	# Layers #7: Softmax 2
	with tf.variable_scope("softmax") as scope:
		softmax = tf.nn.softmax(
			logits = logits,
			name = scope.name
		)

	return logits, softmax
