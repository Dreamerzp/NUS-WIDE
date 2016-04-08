import glob
import numpy


def image_label_read():

    file_count = 0 # Id of image label
    n_images = 269648
    image_label = [-999] * n_images
    ind_label_count = [0] * 81
    labels = list()

    for file in glob.glob("/home/darshan/Downloads/AllLabels/Labels*"):

        parts = file.split("_")
        label = parts[1].replace(".txt", "")
        labels.append(label)
        index = 0
        label_count = 0
        with open(file, 'r') as file_reader:

            for line in file_reader:
                num = int(line.replace("\n", ""))
                if num == 1:
                    image_label[index] = file_count
                    label_count += 1
                index += 1

        ind_label_count[file_count] = label_count
        file_count += 1

    with open("../data/label_ids.txt", "w") as label_writer:

        count = 0
        for label in labels:
            label_writer.write(label + "\t" + str(count) + "\n")
            count += 1

    with open("../data/full_image_label.txt", "w") as label_writer:

        for label in image_label:
            label_str = str(label)
            label_writer.write(label_str + "\n")

    print ind_label_count
    print len(ind_label_count)


def image_mul_label_read():

    file_count = 0 # Id of image label
    n_images = 269648
    image_labels = dict()
    #ind_label_count = [0] * 81
    labels_list = list()
    label_id = 0

    for file in glob.glob("/home/darshan/Downloads/AllLabels/Labels*"):

        parts = file.split("_")
        label = parts[1].replace(".txt", "")
        labels_list.append(label)
        index = 0

        with open(file, 'r') as file_reader:

            for line in file_reader:
                num = int(line.replace("\n", ""))
                if num == 1:
                    if index in image_labels:
                        image_labels[index].append(label_id)
                    else:
                        labels = [label_id]
                        image_labels[index] = labels
                index += 1

        file_count += 1
        label_id += 1

    print 'Total {0} files are explored'.format(file_count)

    with open("../data/mul_label_ids.txt", "w") as label_writer:

        for key, value in image_labels.items():
            label_str = ' '.join(map(str, value))
            label_writer.write(str(key) + "\t" + label_str + "\n")

    with open("../data/label_ids.txt", "w") as label_writer:

        for label, index in enumerate(labels_list):
            label_writer.write(str(label) + "\t" + str(index) + "\n")


def combine_full_image_id_label():
    """
    This function combines the image id and label from two separate files and write in a new file
    :return: None
    """

    labels = list()

    image_ids = list()

    with open("../data/full_image_label.txt", "r") as label_file:

        for line in label_file:
            labels.append(line.replace("\n", ""))

    with open("../data/full_image_id.txt", "r") as id_file:

        for line in id_file:
            image_ids.append(line.replace("\n", ""))

    with open("../data/full_combine_image_id_label.txt", "w") as label_id_file:

        length = len(labels)
        for i in range(length):
            label_id_file.write(image_ids[i] + "\t" + labels[i] + "\n")


def check_missing_labels():
    """
    This function checks the number of missing labels in the given NUS WIDE Image data set
    :return:
    """

    with open("../data/full_combine_image_id_label.txt", "r") as label_id_file:

        count = 0
        for line in label_id_file:

            if line != "":
                parts = line.split("\t")
                if int(parts[1].replace("\n", "")) == -999:
                    count += 1
        print count


def create_available_image_with_label():
    """
    This function compare the downloaded images with the images for which labels are available and creates
    a file with available image ids and labels
    :return:
    """

    image_labels = dict()

    with open("../data/full_combine_image_id_label.txt", "r") as label_id_file:

        for line in label_id_file:

            if line != "":
                parts = line.split("\t")
                if int(parts[1].replace("\n", "")) != -999:
                    image_labels[parts[0]] = parts[1].replace("\n", "")

    good = 0
    bad = 0

    with open("../data/downloaded_image_ids.txt", "r") as downloaded_images:

        with open("../data/available_image_id_label.txt", "w") as available_iid_write:

            for line in downloaded_images:

                if line != "":
                    line = line.strip().replace("\n", "")

                    if line in image_labels:
                        available_iid_write.write(line + "\t" + image_labels[line] + "\n")
                        good += 1
                    else:
                        bad += 1

    print good # 172087
    print bad  # 49649


def check_available_image_with_label():
    """
    This function counts the label in the available images
    :return:
    """

    image_labels = dict()
    label_count = [0] * 81
    count = 0
    with open("../data/available_image_id_label.txt", "r") as available_id_file:

        for line in available_id_file:
            if line != "":
                parts = line.split("\t")
                label = int(parts[1].replace("\n", ""))
                label_count[label] += 1
            count += 1

    print label_count

if __name__ == "__main__":

    image_mul_label_read()



