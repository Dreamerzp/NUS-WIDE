from keras import preprocessing
from PIL import Image
import h5py
import numpy
from scipy.misc import imread
import os


def load_images(image_dir, image_list_file):
    """
    This function loads the images from the given directory
    :param image_dir:
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
                images.append(image)

                counter += 1
                if counter % 5000 == 0:
                    print "{0} number of images are loaded".format(counter)


if __name__ == "__main__":

    image_dir = "/home/darshan/Documents/NUS-WIDE-RESIZE/"
    image_list_file = "../data/available_image_id_label.txt"
    load_images(image_dir, image_list_file)

