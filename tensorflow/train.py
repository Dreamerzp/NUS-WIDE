import os.path
import time
from datetime import datetime

import tensorflow as tf

import image_labels
import image_read
import input
import model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/nus_wide',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def train(file_names):

    with tf.Graph().as_default():

        images, labels = input.inputs(file_names, FLAGS.batch_size)

        # Get images and labels for CIFAR-10.
        #images, labels = cifar10.distorted_inputs()

        global_step = tf.Variable(0, trainable=False)

        logits = model.inference(images)
        #logits = model.cifar10_inference(images)
        #logits = cifar10.inference(images)

        # Calculate loss.
        loss = model.loss(logits, labels)

        train_op = model.train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                                graph_def=sess.graph_def)

        for step in xrange(FLAGS.max_steps):

            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            #assert not numpy.isnan(loss), 'Model diverged with loss = NaN'

            if step % 10 == 0:

                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value,
                         examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 500 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument

    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)

    file_path = "/home/darshan/Documents/NUS-WIDE/data/"
    file_name = "sample_download_image.txt"

    file_names = image_read.file_names(file_path, file_name)
    print('File Names are loaded')

    label_code = image_labels.fetch_label_code()
    print('Labels are loaded')

    labels = image_read.image_labels(label_code)
    #labels = [random.randrange(0, 10, 1) for _ in range(500)]
    updated_file_names = input.updated_filename(file_names, labels)

    #cifar10.maybe_download_and_extract()

    train(updated_file_names)


if __name__ == '__main__':
    tf.app.run()