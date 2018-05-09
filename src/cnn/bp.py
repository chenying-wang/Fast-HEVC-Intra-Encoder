import tensorflow as tf

import util

_fc_with_dropout = util.fc_with_dropout

def bp(features, features_size, keep_prob = 1.0):
	hidden1 = _fc_with_dropout(
		input = features,
		input_size = features_size,
		output_size = 32,
		keep_prob = keep_prob,
		name = "hidden1"
	)
	hidden2 = _fc_with_dropout(
		input = hidden1,
		input_size = 32,
		output_size = 32,
		keep_prob = keep_prob,
		name = "hidden2"
	)
	output = _fc_with_dropout(
		input = hidden2,
		input_size = 32,
		output_size = 1,
		keep_prob = keep_prob,
		name = "output"
	)

	return output

if __name__ == "__main__":
	tf.app.run()
