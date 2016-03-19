import Image
import random

import imageflow
import numpy

import image_read


def PIL2array(img):

    return numpy.array(img.getdata(), numpy.uint8).reshape(-1, img.size[1], img.size[0],  3)


def convert_into_tf_record(file_names, labels):

    images = list()
    labels = numpy.array(labels)

    for file_name in file_names:
        image = Image.open(file_name)
        image = PIL2array(image)
        images.append(image)

    images = numpy.array(images)

    print "Loaded all the images"
    imageflow.convert_images(images, labels, "dummy_cifar_data")

if __name__ == "__main__":

    file_path = "/home/darshan/Documents/NUS-WIDE/data/"
    file_name = "images.txt"

    file_names = image_read.file_names(file_path, file_name)
    print('File Names are loaded')

    #label_code = image_labels.fetch_label_code()
    #print('Labels are loaded')

    #labels = image_read.image_labels(label_code)

    #labels = input.partial_labels(file_names, labels)

    print  len(file_names)
    labels = [random.randrange(0, 10, 1) for _ in range(499)]

    print len(labels)

    convert_into_tf_record(file_names, labels)