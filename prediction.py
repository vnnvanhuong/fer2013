from keras.models import load_model
from utils import load_test_data, plot_prediction
from keras.utils import to_categorical


model = load_model('model.h5')

x_test, y_test = load_test_data()
data_for_prediction = x_test.reshape((x_test.shape[0], 48, 48, 1)).astype('float32')
y_test = to_categorical(y_test, 7)

predictions = model.predict(data_for_prediction)

plot_prediction(predictions, y_test, x_test)