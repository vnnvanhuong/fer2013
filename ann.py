from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.losses import mean_squared_error, categorical_crossentropy
from keras.optimizers import SGD
from keras.utils import to_categorical
import numpy as np

from utils import load_data, plot_dataset, plot_training_history

# read dataset and split it into input set and labels set
x_train, y_train = load_data();
print(x_train.shape)
print(y_train.shape)

# plot the dataset
# plot_dataset(x_train, y_train)

# pre-process
n_train=20000
x_train, x_test = x_train[:n_train, :], x_train[n_train:,:]
y_train, y_test = y_train[:n_train], y_train[n_train:]

y_train, y_test = to_categorical(y_train, 7), to_categorical(y_test, 7)

# create model
model = Sequential()
model.add(Flatten(input_shape=(48,48)))
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=7, activation="softmax"))

model.compile(optimizer=SGD(lr=0.001), loss=mean_squared_error, metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=500, epochs=20, validation_data=(x_test, y_test))

plot_training_history(history)
	
