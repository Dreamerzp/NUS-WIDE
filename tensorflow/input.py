import tensorflow as tf

import model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_epochs', 100,
                            """Number of epochs""")

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
IMAGE_SIZE = 224
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 500
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 100


def inputs(file_names, batch_size):

    #partial_labels = tf.cast(partial_labels, dtype=tf.int32)
    #print type(partial_labels)

    '''
    images, labels = imageflow.distorted_inputs(
        filename="/home/darshan/Documents/NUS-WIDE/converted_data/dummy_cifar_data.tfrecords",
        batch_size=FLAGS.batch_size,
        num_epochs=FLAGS.num_epochs,
        num_threads=5, imshape=[32, 32, 3], imsize=224)

    print labels
    images = tf.reshape(images, [128, IMAGE_SIZE, IMAGE_SIZE, 3])
    print images
    return images, labels

    '''

    read_input = read_image_data(file_names)

    min_fraction_of_examples_in_queue = 0.01
    num_examples_per_epoch = model.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(read_input.images,
                                           read_input.labels,
                                           min_queue_examples,
                                           batch_size,
                                           shuffle=False)


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):

    num_preprocess_threads = 16

    if not shuffle:
        images, label_batch = tf.train.batch([image, label],
                                             batch_size=batch_size,
                                             num_threads=num_preprocess_threads,
                                             capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.image_summary('images', images)

    return images, tf.reshape(label_batch, [batch_size])


def read_image_data(file_names):

    class Record(object):
        pass
    result = Record()

    #lv = tf.cast(labels, tf.int32)

    filename_queue = tf.train.string_input_producer(file_names)

    filename, labels = tf.decode_csv(filename_queue.dequeue(), [[""], [""]], " ")
    #reader = tf.WholeFileReader()
    file_contents = tf.read_file(filename)

    images = tf.image.decode_jpeg(file_contents, channels=3)

    float_images = tf.cast(images, tf.float32)
    float_images.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])

    result.labels = labels
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


def updated_filename(file_names, labels):

    updated_filenames = list()

    for filename in file_names:

        parts = filename.split('/')
        only_filename = parts[-1]
        label = labels[only_filename]
        updated_filenames.append(filename +' ' + str(label))

    return updated_filenames


def diff_updated_filename(file_names, labels):

    updated_filenames = list()

    for i in range(len(file_names)):

        filename = file_names[i]
        parts = filename.split('/')
        only_filename = parts[-1]
        updated_filenames.append(filename +' ' + str(labels[i]))

    return updated_filenames