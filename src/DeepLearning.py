import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow.python.keras.callbacks import EarlyStopping

data_path = './../data/UniversalBank_Final.csv'
data = pd.read_csv(data_path)

# dependent and independents variables
y = data['Personal_Loan']
X = data.drop(columns=['Personal_Loan'])

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from numpy.random import seed

seed(1)
tf.random.set_seed(1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

target_train = to_categorical(y_train)
n_cols = X_train.shape[1]

print(y_train[0])
print(target_train[0])
print('n_cols:', n_cols)

model = Sequential()
model.add(Dense(20, activation='relu', input_shape=(n_cols,)))
model.add(Dense(2, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, target_train,
          validation_data=(X_test, to_categorical(y_test)),
          verbose=0)

y_pred = (model.predict(X_test) > 0.5).astype("int32")[:, 1]
con_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_pred).numpy()
print(con_mat)
