import tensorflow as tf

import dataset
import cnn

TRAINING_LOG_DIR = "./.tmp/train"
EPOCH = 1
SAMPLE_SIZE = 51520000
BATCH_SIZE = 64

CKPT_PATH = "./.tmp/training.ckpt"
CKPT_STEP = 1000

def train():
	with tf.variable_scope("input") as scope:
		iterator = dataset.get_train().batch(BATCH_SIZE).repeat(EPOCH).make_initializable_iterator()
		features, labels = iterator.get_next()
		onehot_labels = tf.one_hot(indices = labels, depth = 2, dtype = tf.int64)

	logits, softmax = cnn.inference(features, dataset.FEATURE_WIDTH, dataset.FEATURE_HEIGHT)
	pred = tf.argmax(logits, 1)

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
		labels = onehot_labels,
		logits = logits
	))
	
	accuracy = tf.reduce_mean(tf.cast(
		tf.equal(pred, labels)
	, tf.float32))

	optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
	train_batch = optimizer.minimize(loss)
	
	tf.summary.scalar("loss", loss, family = "train")
	tf.summary.scalar("accuracy", accuracy, family = "train")
	merged = tf.summary.merge_all()

	summary_writer = tf.summary.FileWriter(TRAINING_LOG_DIR, tf.get_default_graph())

	saver = tf.train.Saver()

	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		iterator.initializer.run()

		if tf.train.get_checkpoint_state(CKPT_PATH):
			saver.restore(sess, tf.train.latest_checkpoint(CKPT_PATH))

		for step in range (int(EPOCH * SAMPLE_SIZE / BATCH_SIZE)):
			_, avg_loss, acc = sess.run([train_batch, loss, accuracy])
			print("Step %d, loss = %f, accuracy = %.2f%%" % (step, avg_loss, 100 * acc))
			summary = sess.run(merged)
			summary_writer.add_summary(summary, step)
			
			if (step + 1) % CKPT_STEP == 0:
				saver.save(
					sess = sess,
					save_path = CKPT_PATH,
					global_step = step
				)
				print("Model Saved")
		print("Training Done")
		
	summary_writer.close()

def main(unused_argv):
	train()

if __name__ == "__main__":
	tf.app.run()
