from keras import preprocessing
from PIL import Image
import h5py
import numpy
from scipy.misc import imread
import os


def write_train_npy(image_dir, image_list_file):
    """
    This function reads the train image files and converts them into npy file
    :param image_dir:
    :param image_list_file:
    :return:
    """

    image_label = dict()
    with open(image_list_file, "r") as image_ids_reader:

        for line in image_ids_reader:
            image_id_label = line.replace("\n", "")
            parts = image_id_label.split("\t")
            image_label[parts[0]] = int(parts[1])

            '''
            image_path = image_dir + "/" + parts[0] + ".jpg"
            labels.append(int(parts[1]))
            image = imread(image_path)
            images.append(image)
            counter += 1
            if counter % 10000 == 0:
                print "{0} number of images are loaded".format(counter)
            '''

    image_ids = list(image_label)
    images = list()
    labels = list()
    print "Image Ids are loaded"

    counter = 0

    for subdir, _, files in os.walk(image_dir):

        for file in files:

            parts = file.split("/")
            file_name = parts[-1].replace(".jpg", "")

            if file_name in image_ids:

                image_path = os.path.join(subdir, file)
                #image = Image.open(image_path)
                image = imread(image_path)
                #image = image.astype(numpy.float)
                images.append(image)
                labels.append(image_label[file_name])

                counter += 1
                if counter % 5000 == 0:
                    print "{0} number of images are loaded".format(counter)

    images = numpy.array(images)
    labels = numpy.array(labels)

    h5f = h5py.File('../data/Sample.h5', 'w')
    h5f.create_dataset('features', data=images)
    h5f.create_dataset('targets', data=labels)
    h5f.close()


def read_file_names(image_dir, file_name):

    file_names = list()

    with open(file_name) as reader:

        for line in reader:
            if line != '':
                parts = line.split("\t")
                image_id = parts[0]
                file_names.append(image_dir+image_id+'.jpg')

    return file_names


def read_image_labels(file_name):

    labels = list()

    with open(file_name) as reader:

        for line in reader:
            if line != '':
                parts = line.split("\t")
                label = parts[1].replace("\n", "")
                labels.append(label)

    labels = numpy.array(labels)
    return labels


if __name__ == "__main__":

    image_dir = "/home/darshan/Documents/NUS-WIDE-RESIZE/"
    image_list_file = "../data/sample_train_available_image_id_label.txt"

    file_names = read_file_names(image_dir, image_list_file)

    #write_train_npy(image_dir, image_list_file)