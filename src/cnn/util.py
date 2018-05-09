import tensorflow as tf

def activate(features, name = None):

	return tf.nn.leaky_relu(
		features = features,
		alpha = 0.01,
		name = name
	)


def conv2d(input,
		   input_shape,
		   filters,
		   filter_size,
		   strides = 1,
		   biases = True,
		   activation = False,
		   name = None):

	with tf.variable_scope(name):
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
			output = activate(pre_activation)
		else:
			output = pre_activation
	
	return output


def max_pooling(input,
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


def avg_pooling(input,
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


def fc_with_dropout(input,
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
		output = activate(
			features = tf.matmul(input_dropout, weights) + biases,
			name = scope.name
		)
	
	return output


def inception(input,
			  width,
			  height,
			  input_channel,
			  branch_channel,
			  reduced_channel,
			  name = None):

	with tf.variable_scope(name) as scope:
		branch0 = conv2d(
			input = input,
			input_shape = [width, height, input_channel],
			filters = branch_channel[0],
			filter_size = [1, 1],
			biases = True,
			activation = True,
			name = "1x1"
		)
		branch1 = conv2d(
			input = conv2d(
				input = input,
				input_shape = [width, height, input_channel],
				filters = reduced_channel[1],
				filter_size = [1, 1],
				biases = True,
				activation = True,
				name = "3x3_reduced"
			),
			input_shape = [width, height, reduced_channel[1]],
			filters = branch_channel[1],
			filter_size = [3, 3],
			biases = True,
			activation = True,
			name = "3x3"
		)
		branch2 = conv2d(
			input = conv2d(
				input = conv2d(
					input = input,
					input_shape = [width, height, input_channel],
					filters = reduced_channel[1],
					filter_size = [1, 1],
					biases = True,
					activation = True,
					name = "5x5_reduced"
				),
				input_shape = [width, height, reduced_channel[1]],
				filters = branch_channel[2],
				filter_size = [3, 3],
				biases = True,
				activation = True,
				name = "5x5_pre"
			),
			input_shape = [width, height, branch_channel[2]],
			filters = branch_channel[2],
			filter_size = [3, 3],
			biases = True,
			activation = True,
			name = "5x5"
		)
		branch3 = conv2d(
			input = max_pooling(
				input = input,
				pool_size = 3,
				strides = 1,
				name = "pool"
			),
			input_shape = [width, height, input_channel],
			filters = branch_channel[3],
			filter_size = [1, 1],
			biases = True,
			activation = True,
			name = "pool_reduced"
		)
		output = tf.concat(
			values = [branch0, branch1, branch2, branch3],
			axis = -1,
			name = "concat"
		)

	return output


def inception_64x64(input,
					input_channel,
					branch_channel,
					reduced_channel,
					name = None):

	return inception(
		input = input,
		width = 64,
		height = 64,
		input_channel = input_channel,
		branch_channel = branch_channel,
		reduced_channel = reduced_channel,
		name = name
	)


def inception_32x32(input,
					input_channel,
					branch_channel,
					reduced_channel,
					name = None):

	return inception(
		input = input,
		width = 32,
		height = 32,
		input_channel = input_channel,
		branch_channel = branch_channel,
		reduced_channel = reduced_channel,
		name = name
	)


def inception_16x16(input,
					input_channel,
					branch_channel,
					reduced_channel,
					name = None):

	return inception(
		input = input,
		width = 16,
		height = 16,
		input_channel = input_channel,
		branch_channel = branch_channel,
		reduced_channel = reduced_channel,
		name = name
	)


def inception_8x8(input,
				  input_channel,
				  branch_channel,
				  reduced_channel,
				  name = None):

	return inception(
		input = input,
		width = 8,
		height = 8,
		input_channel = input_channel,
		branch_channel = branch_channel,
		reduced_channel = reduced_channel,
		name = name
	)
