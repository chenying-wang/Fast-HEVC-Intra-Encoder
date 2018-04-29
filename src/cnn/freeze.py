import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph

import train

STEPS = 229000

INPUT_GRAPH = train.CKPT_PATH + train.GRAPH_FILENAME
INPUT_CKPT = train.CKPT_PATH + train.CKPT_PREFIX + "-" + str(STEPS)

OUTPUT_GRAPH = train.CKPT_PATH + "frozen_graph.pb"
OUTPUT_NODE = "softmax/softmax"

def freeze():
	freeze_graph(input_graph = INPUT_GRAPH,
        input_saver = "",
        input_binary = False,
		input_checkpoint = INPUT_CKPT,
		output_node_names = OUTPUT_NODE,
		restore_op_name = "DEPRECATED",
		filename_tensor_name = "DEPRECATED",
		output_graph = OUTPUT_GRAPH,
		clear_devices = True,
		initializer_nodes = "")

def main(unused_argv):
	freeze()

if __name__ == "__main__":
	tf.app.run()
 