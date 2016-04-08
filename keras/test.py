import h5py
from scipy.misc import imread
import data


def create_hdf5(image_dir, image_list_file):

    file_names = data.read_file_names(image_dir, image_list_file)
    labels_r = data.read_image_labels(image_list_file)
    print labels_r.shape
    f = h5py.File('../data/Sample_Train.hdf5')  # make an hdf5 file

    features = f.require_dataset('/features', shape=(len(file_names), 224, 224, 3), dtype=int)
    f.create_dataset('/labels', data=labels_r)

    count = 0
    for i, fn in enumerate(file_names):
        im = imread(fn)
        features[i, :, :, :] = im
        if count % 200 == 0:
            print "{0} images are saved".format(count)
        count += 1

    f.close()


def read_hdf5(file_path):

    h5f = h5py.File(file_path, 'r')
    images = h5f['features']
    labels = h5f['labels']
    print (type(images))
    print (images.shape)
    print (type(labels))
    print (labels.shape)

    h5f.close()

if __name__ == "__main__":

    image_dir = "/home/darshan/Documents/NUS-WIDE-RESIZE/"
    image_list_file = "../data/sample_train_available_image_id_label.txt"
    create_hdf5(image_dir, image_list_file)
    read_hdf5('../data/Sample_Train.hdf5')

