import numpy
import model
import sys
from getopt import getopt,GetoptError
from keras.utils.io_utils import HDF5Matrix
import h5py


def load_train_data(data_path, train_start, n_training_examples):
    """
    Load training data from .npy files.
    """

    '''
    X = numpy.load('../data/Sample_Images_train.npy')
    y = numpy.load('../data/Sample_Labels_train.npy')

    #seed = np.random.randint(1, 10e6)
    seed = 124
    numpy.random.seed(seed)
    numpy.random.shuffle(X)
    numpy.random.seed(seed)
    numpy.random.shuffle(y)
    '''
    '''
    X_train = HDF5Matrix(data_path, 'features', train_start, train_start + n_training_examples)
    y_train = HDF5Matrix(data_path, 'targets', train_start, train_start + n_training_examples)

    return X_train, y_train
    '''

    h5f = h5py.File(data_path, 'r')
    images = h5f['features']
    labels = h5f['labels']
    return images, labels
    h5f.close()


def split_data(X, y, split_ratio=0.2):
    """
    Split data into training and testing.

    :param X: X
    :param y: y
    :param split_ratio: split ratio for train and test data
    """
    split = X.shape[0] * split_ratio
    X_test = X[:split, :, :, :]
    y_test = y[:split, :]
    X_train = X[split:, :, :, :]
    y_train = y[split:, :]

    return X_train, y_train, X_test, y_test


def train():
    """
    param: n_iter -> iteration to start from
    Training systole and diastole models.
    """

    data_path = '../data/Sample_Train.hdf5'

    print('Loading and compiling models...')
    alexnet_model = model.alexnet_model()

    print('Loading training data...')
    X, y = load_train_data(data_path, 0, 20000)

    print X.shape
    print y.shape
    print('Training Data Loaded')

    nb_iter = 10
    epochs_per_iter = 1
    batch_size = 16
    calc_crps = 0  # calculate CRPS every n-th iteration (set to 0 if CRPS estimation is not needed)

    # remember min val. losses (best iterations), used as sigmas for submission
    min_val_loss_systole = sys.float_info.max
    min_val_loss_diastole = sys.float_info.max

    n_iter = 0

    print('-'*50)
    print('Training...')
    print('-'*50)

    i = n_iter
    while (i < nb_iter):

        print('-'*50)
        print('Iteration {0}/{1}'.format(i + 1, nb_iter))
        print('-'*50)

        print('Fitting Alexnet model...')
        cross_entropy = alexnet_model.fit(X, y, nb_epoch=epochs_per_iter,
                                          batch_size=batch_size, shuffle="batch")

        # sigmas for predicted data, actually loss function values (RMSE)
        loss_systole = cross_entropy.history['loss'][-1]
        #val_loss_systole = cross_entropy.history['val_loss'][-1]
        print type(loss_systole)

        print('Saving weights...')
        # save weights so they can be loaded later
        alexnet_model.save_weights('weights_alexnet.hdf5', overwrite=True)

        '''
        # for best (lowest) val losses, save weights
        if val_loss_systole < min_val_loss_systole:
            min_val_loss_systole = val_loss_systole
            alexnet_model.save_weights('weights_alexnet_best.hdf5', overwrite=True)

        '''

        # save best (lowest) val losses in file (to be later used for generating submission)
        with open('../data/val_loss.txt', mode='w+') as f:
            f.write(str(min_val_loss_systole))
            f.write('\n')
            f.write(str(min_val_loss_diastole))
            f.write('\n')
            f.write(str(i))
            f.write('\n')

        i += 1

if __name__ == '__main__':

    train()