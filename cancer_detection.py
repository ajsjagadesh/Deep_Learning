import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.utils import column_or_1d
from keras.models import Sequential
from keras.layers import Dense


df = pd.read_csv('data.csv')  
df = df.drop(columns=['Unnamed: 32', 'id'], axis='columns')
df['diagnosis'].replace(['M', 'B'], [0,1], inplace=True)

X = df.drop(columns=['diagnosis'], axis='columns')
Y = df['diagnosis']

df['diagnosis'].value_counts().plot(kind='bar')

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

model = Sequential()
model.add(Dense(40, input_dim=30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=10)

from keras.metrics import accuracy
_, accuracy = model.evaluate(X_test, y_test)
print(f"accuracy : {(accuracy * 100)}")

pred = model.predict(X_test)
#print(pred)
v = []
for i in range(len(pred)):
  if pred[i] > 0.5:
    v.append(1)
  if pred[i] < 0.5:
    v.append(0)

v = pd.Series(v)

accuracy = accuracy_score(y_test,v)
print(accuracy)