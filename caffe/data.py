import numpy as np


def create_input_caffe_file(image_source_dir, source_image_label_file, target_image_label_file):

    file_paths_label = dict()

    with open('../data/'+source_image_label_file, 'r') as file_reader:
        for line in file_reader:

            if line != "":
                parts = line.replace("\n","").split("\t")
                file_path = image_source_dir + parts[0] + ".jpg"
                file_paths_label[file_path] = parts[1]

    print len(file_paths_label)

    with open('../data/'+target_image_label_file, 'w') as file_writer:
        for key, value in file_paths_label.iteritems():
            file_writer.write(key+"\t"+value+"\n")

if __name__ == "__main__":

    create_input_caffe_file("/home/darshan/Documents/NUS-WIDE-RESIZE/", "test_image_id_label.txt",
                            "final_test_image_id_label.txt")