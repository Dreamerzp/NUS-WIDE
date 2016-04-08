def image_label():

    file_path = "/home/darshan/Documents/NUS-WIDE/data/"
    file_name = "imagelist.txt"

    filename_label = dict()
    label_dict = dict()

    with open(file_path+file_name) as reader:
        for line in reader:
            if line != '':
                line = line.replace(".jpg", "").strip()
                parts = line.split('\\')

                label_dict[parts[0]] = parts[0]

                filename_parts = parts[1].split('_')

                filename_label[filename_parts[1]] = parts[0]

    print 'Unique Labels : ' + str(len(label_dict))
    print 'Number of records : ' + str(len(filename_label))

    with open("/home/darshan/Documents/NUS-WIDE/data/image_labels.txt", "w") as file:
        for key, value in filename_label.viewitems():
            file.write(key+' '+value+"\n")

    return filename_label


def create_label_code():

    unique_labels = set()

    with open("/home/darshan/Documents/NUS-WIDE/data/image_labels.txt", "r") as file:

        for line in file:
            parts = line.split(' ')
            unique_labels.add(parts[1].replace("\n", ""))

    with open("/home/darshan/Documents/NUS-WIDE/data/label_code.txt", "w") as file:

        count = 0
        for label in unique_labels:
            file.write(label+' '+str(count)+'\n')
            count += 1


def fetch_label_code():

    label_code = dict()
    with open("/home/darshan/Documents/NUS-WIDE/data/label_code.txt", "r") as file:

        for line in file:
            parts = line.split(' ')
            label_code[parts[0]] = int(parts[1].replace("\n", ""))

    return label_code


def read_image_label():

    image_labels = dict()
    unique_labels = set()

    with open("/home/darshan/Documents/NUS-WIDE/data/image_labels.txt", "r") as file:

        for line in file:
            parts = line.split(' ')
            image_labels[parts[0]] = parts[1]
            unique_labels.add(parts[1])

    print 'number of labels ' + str(len(unique_labels))

if __name__ == "__main__":

    create_label_code()
    fetch_label_code()


