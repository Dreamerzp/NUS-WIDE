import tensorflow as tf


def file_names(file_path, file_name):

    filename_list = list()

    default_path = "/home/darshan/Documents/NUS-WIDE-RESIZE/"

    with open(file_path+file_name) as reader:

        for line in reader:
            if line != '':
                filename_list.append(default_path+str(line.replace("\n", ""))+'.jpg')

    return filename_list


def filenames(default_path):

    file_path = "/home/darshan/Documents/NUS-WIDE/data/"
    file_name = "downloaded_image_ids.txt"

    filenames = list()

    with open(file_path+file_name) as reader:

        for line in reader:
            if line != '':
                filenames.append(default_path+str(line)+'.jpg')

    return filenames


def image_labels(label_code):

    file_path = "/home/darshan/Documents/NUS-WIDE/data/"
    file_name = "image_labels.txt"

    image_labels = dict()

    with open(file_path + file_name) as reader:

        for line in reader:
            if line != '':
                parts = line.split(' ')
                filename = parts[0] + str('.jpg')
                label = parts[1].replace("\n", "")
                image_labels[filename] = label_code[label]

    return image_labels


def image_label_list(filenames, image_labels):

    for name in filenames:
        name = name.replace(".jpg", "")
        parts = name.split('/')
        filename = parts[-1]


if __name__ == "__main__":

    default_path = "/Users/Darshan/Documents/kaggle/NUS-WIDE_IMAGES/"
    filenames = filenames(default_path)
    print filenames

    #image_labels = image_labels()