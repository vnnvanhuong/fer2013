import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data():
	train = pd.read_csv('fer2013.csv', delimiter=',')

	y_train = train['emotion'].to_numpy()
	x_train = np.zeros((y_train.shape[0], 48,48), dtype=np.float32)
	
	for i,row in train.iterrows():
		pixels=np.asarray(list(row['pixels'].split(' ')),dtype=np.float32)
		x_train[i] = pixels.reshape((48,48))
	
	return x_train, y_train

def load_test_data():
	train = pd.read_csv('train.csv', delimiter=',')

	y_train = train['emotion'].to_numpy()
	x_train = np.zeros((y_train.shape[0], 48,48), dtype=np.float32)

	for i,row in train.iterrows():
		pixels=np.asarray(list(row['pixels'].split(' ')),dtype=np.float32)
		x_train[i] = pixels.reshape((48,48))

	return x_train, y_train

def plot_dataset(x_train, y_train):
	emotions = { 0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral' }
	plt.figure(figsize=(6,7.5))
	for i in range(25):
		plt.subplot(5, 5, i+1)
		plt.xticks([])
		plt.yticks([])
		plt.xlabel(emotions[y_train[i]])
		plt.imshow(x_train[i])
	plt.show()

def plot_training_history(history):
  plt.subplot(2,1,1)
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='lower right')

  plt.subplot(2,1,2)
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])

  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper right')

  plt.tight_layout()

  plt.show()

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img)
  predicted_label = np.argmax(predictions_array)
  if (predicted_label == true_label).all():
    color = 'blue'
  else:
    color = 'red'

  # true_label is in one hot encoding format
  true_label_index = np.where(true_label == 1)[0][0]

  class_names = { 0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral' }

  plt.xlabel("{} -- {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label_index]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(7))
  plt.yticks([])
  thisplot = plt.bar(range(7), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  # true_label is in one hot encoding format
  true_label_index = np.where(true_label == 1)[0][0]

  thisplot[predicted_label].set_color('red')
  thisplot[true_label_index].set_color('blue')


def plot_prediction(predictions, y_test, x_test):
  num_rows = 5
  num_cols = 5
  num_images = num_rows*num_cols
  plt.figure(figsize=(2*2*num_cols, 2*num_rows))
  for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], y_test, x_test)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], y_test)
  plt.tight_layout()
  plt.show()