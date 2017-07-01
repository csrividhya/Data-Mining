import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras.backend as K
from sklearn.metrics import *

def f_measure(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label='1')

# Generate dummy data
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))

model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)

score = model.evaluate(x_test, y_test, batch_size=128)
