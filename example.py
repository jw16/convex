import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D, Flatten


# vector input
input = Input(shape=(200,1))
# fully connected layer w/ 200 nodes
layer1 = Conv1D(5, 20,activation="sigmoid")(input)
layer2 = MaxPooling1D(5)(layer1)
layer3 = Flatten()(layer2)
# binary 'classifier' layer
output = Dense(1, activation="sigmoid")(layer3)

# set the input and output layers
model = Model(inputs=input, outputs=output)
model.summary() # print layer info

# tell tensorflow what optimizer/loss function
model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics=['accuracy'])

# now ready for data
import h5py
import numpy as np
data = h5py.File("data.h5", 'r')

allX = data['X'] # list of 200-sample vectors
allY = data['Y'] # list of 0,1 'labels'

trainX = []; testX = []
trainY = []; testY = []

N = len(allX)
indexes = list(range(N))
np.random.shuffle(indexes)

M = int(N* 0.2) # 80/20 train/test split

for idx in indexes[0:M] :
    testX.append(allX[idx])
    testY.append(allY[idx])


for idx in indexes[M:] :
    trainX.append(allX[idx])
    trainY.append(allY[idx])


# ok, now we're ready to train.
trainX = np.array(trainX)
trainY = np.array(trainY)
testX = np.array(testX)
testY = np.array(testY)

model.fit(trainX, trainY, epochs=40, batch_size=64,
          validation_data=(testX, testY))

# here we go.

results = model.evaluate(testX, testY, batch_size=128)
print("test loss, test acc:", results)
