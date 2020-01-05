import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data():
	train = pd.read_csv('train.csv', delimiter=',')

	y_train = train['emotion'].to_numpy()
	x_train = np.zeros((y_train.shape[0], 48,48), dtype=np.uint8)
	
	for i,row in train.iterrows():
		pixels=np.asarray(list(row['pixels'].split(' ')),dtype=np.uint8)
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
  fig = plt.figure()
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