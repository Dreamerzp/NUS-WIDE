import tensorflow as tf

import image_labels
import image_read


def read_image_data(file_names, labels):

    class Record(object):
        pass
    result = Record()

    lv = tf.constant(labels)
    label_fifo = tf.FIFOQueue(len(file_names), tf.int32, shapes=[[]])

    file_fifo = tf.train.string_input_producer(file_names,
                                               shuffle=False,
                                               capacity=len(file_names))
    label_enqueue = label_fifo.enqueue_many([lv])

    filename_queue = tf.train.string_input_producer(file_names, shuffle=True)

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    images = tf.image.decode_jpeg(value, channels=3)
    float_images = tf.cast(images, tf.float32)
    float_images.set_shape([224, 224, 3])

    uint8image = float_images
    label = label_fifo.dequeue()

    min_fraction_of_examples_in_queue = 0.4
    num_examples_per_epoch = 10
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)
    num_preprocess_threads = 2

    shuffle_batch = tf.train.shuffle_batch([uint8image, label],
                                           batch_size=2,
                                           num_threads=num_preprocess_threads,
                                           capacity=min_queue_examples + 3 * 2,
                                           min_after_dequeue=min_queue_examples)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        tf.train.start_queue_runners(sess=sess)
        sess.run(shuffle_batch)

        for i in xrange(2):
            value = sess.run(label)
            print "F: ", value


def partial_labels(file_names, labels):

    partial_labels = list()

    for filename in file_names:
        parts = filename.split('/')
        filename = parts[-1].replace("\n", "")
        partial_labels.append(labels[filename])

    return partial_labels

if __name__ == "__main__":

    file_path = "/Users/Darshan/Documents/kaggle/NUS-WIDE/"
    file_name = "downloaded_image_ids_sample.txt"

    file_names = image_read.file_names(file_path, file_name)
    print('File Names are loaded')
    label_code = image_labels.fetch_label_code()
    print('Labels are loaded')
    labels = image_read.image_labels(label_code)

    partial_labels = partial_labels(file_names, labels)

    read_image_data(file_names, partial_labels)