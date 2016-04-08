from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
shuffle = np.arange(len(iris.data))
np.random.shuffle(shuffle)
iris.data = iris.data[shuffle]
iris.target = iris.target[shuffle]

print iris.data.shape
print iris.target.shape

model = Sequential()
model.add(Dense(3, init='uniform'))
model.add(Activation('softmax'))

model.compile(loss='mean_squared_error', optimizer='rmsprop')

labels = np_utils.to_categorical(iris.target)
model.fit(iris.data, labels, nb_epoch=5, batch_size=1, show_accuracy=True, validation_split=0.3)