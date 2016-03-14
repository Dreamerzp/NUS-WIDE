import tensorflow as tf
import image_read
import image_labels
from datetime import datetime
import os.path
import time
import Image
import numpy
import model
IMAGE_SIZE = 224


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def read_image_data(file_names, labels):

    class Record(object):
        pass
    result = Record()

    #lv = tf.cast(labels, tf.int32)

    filename_queue = tf.train.string_input_producer(file_names)

    filename, label = tf.decode_csv(filename_queue.dequeue(), [[""], [""]], " ")
    #reader = tf.WholeFileReader()
    file_contents = tf.read_file(filename)

    images = tf.image.decode_jpeg(file_contents, channels=3)

    float_images = tf.cast(images, tf.float32)
    float_images.set_shape([224, 224, 3])

    result.labels = label
    result.images = float_images

    return result


def partial_labels(file_names, labels):

    partial_labels = list()

    for filename in file_names:
        parts = filename.split('/')
        filename = parts[-1].replace("\n", "")
        partial_labels.append(labels[filename])

    return partial_labels


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def inputs(file_names, partial_labels):

    #partial_labels = tf.cast(partial_labels, dtype=tf.int32)

    print type(partial_labels)
    read_input = read_image_data(file_names, partial_labels)

    min_fraction_of_examples_in_queue = 0.01
    num_examples_per_epoch = model.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(read_input.images, read_input.labels,
                                           min_queue_examples,
                                           FLAGS.batch_size)


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size):

    num_preprocess_threads = 16

    images, label_batch = tf.train.batch([image, label],
                                         batch_size=batch_size,
                                         num_threads=num_preprocess_threads,
                                         capacity=min_queue_examples + 3 * batch_size,)

    # Display the training images in the visualizer.
    tf.image_summary('images', images)

    return images, tf.reshape(label_batch, [batch_size])


def train(file_names, partial_labels):

    with tf.Graph().as_default():

        images, labels = inputs(file_names, partial_labels)

        global_step = tf.Variable(0, trainable=False)

        logits, parameters = model.inference(images)

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

        assert not numpy.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 10 == 0:
            num_examples_per_step = FLAGS.batch_size
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)

            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print (format_str % (datetime.now(), step, loss_value,
                     examples_per_sec, sec_per_batch))

        if step % 100 == 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)


def update_filename(file_names, labels):

    updated_filenames = list()
    for filename in file_names:

        parts = filename.split('/')
        only_filename = parts[-1]
        label = labels[only_filename]
        updated_filenames.append(filename +' ' +str(label))

    return updated_filenames

if __name__ == "__main__":

    file_path = "/Users/Darshan/Documents/kaggle/NUS-WIDE/"
    file_name = "downloaded_image_ids.txt"

    file_names = image_read.file_names(file_path, file_name)
    print('File Names are loaded')

    label_code = image_labels.fetch_label_code()
    print('Labels are loaded')

    labels = image_read.image_labels(label_code)

    updated_filenames = update_filename(file_names, labels)

    partial_labels = partial_labels(file_names, labels)

    train(updated_filenames, partial_labels)