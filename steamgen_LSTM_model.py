import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed


class model:

    def __init__(self):
        self.model = Sequential()

        self.model.add(LSTM(100,
                            activation='relu',
                            input_shape=(2, 1)))
        self.model.add(RepeatVector(1))
        self.model.add(LSTM(100, activation='relu', return_sequences=True))
        self.model.add(TimeDistributed(Dense(1)))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X_train, y_train, epochs):
        self.model.fit(X_train, y_train, epochs=epochs, validation_split=0.3, verbose=1)

    def plot_model_fit(self):
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.plot(self.model.history.history['loss'], 'b', label='training')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')

        plt.subplot(122)
        plt.plot(self.model.history.history['val_loss'], 'r', label='testing')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)

        return y_pred

    def plot_predict_results(self,y_test,y_pred):
        rmse = np.sqrt(mean_squared_error(y_test.reshape(1, -1), y_pred.reshape(1, -1)))

        print('RMSE is ', rmse)

        plt.figure()
        plt.plot(y_test, 'b', label='actual')
        plt.plot(y_pred, 'r', linestyle='dashed', alpha=0.5, label='prediction')
        plt.show()




