import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import cv2 as cv


print("Hello")

json_file = open('model.json', 'r')

loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")

loaded_model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


img = cv.imread('model/images/blank.png', 0)
X_predict = []

gray = cv.resize(img, (28,28), interpolation=cv.INTER_NEAREST)

cv.imshow('gray',gray)

X_predict.append(gray)

X_predict = np.array(X_predict)

X_predict = np.reshape(X_predict, X_predict.shape+(1,))

predict = loaded_model.predict_classes(X_predict)
print(predict)


print("Hello2")
