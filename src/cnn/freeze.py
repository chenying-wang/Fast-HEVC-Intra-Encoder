import sys
import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph

import train

def freeze(size_index, steps):
	freeze_graph(
		input_graph = train.CKPT_PATH(size_index) + train.GRAPH_FILENAME,
        input_saver = "",
        input_binary = False,
		input_checkpoint = train.CKPT_PATH(size_index) + train.CKPT_PREFIX + "-" + str(steps),
		output_node_names = "softmax/softmax",
		restore_op_name = "DEPRECATED",
		filename_tensor_name = "DEPRECATED",
		output_graph = train.CKPT_PATH(size_index) + "frozen_graph.pb",
		clear_devices = True,
		initializer_nodes = ""
	)

def main(unused_argv):
	freeze(
		size_index = int(sys.argv[1]),
		steps = int(sys.argv[2])
	)

if __name__ == "__main__":
	tf.app.run()
 