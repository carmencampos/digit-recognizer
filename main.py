
# Import required packages
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

# Get train and test data into dataframes
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

# Info
print(df_train.info())
# Describe
print(df_train.describe())
# Show first few rows
print(df_train.head())
# Show column names
print(df_train.columns)

# Load datasets
aux1 = df_train.drop('label', axis=1)
train_X = df_train[aux1.columns]
train_Y = df_train[['label']]
test_X = df_test[aux1.columns]
print(train_X)
print(train_Y)
print(test_X)

# Plot first few images
for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(np.array(train_X.iloc[i]).reshape(28,28), cmap=plt.get_cmap('gray'))
# show the figure
plt.show()

# Reshape images
print(train_X.shape)
train_X = np.array(train_X).reshape(42000, 28, 28)
print(train_X.shape)
test_X = np.array(test_X).reshape(28000, 28, 28)

plt.imshow(train_X[10], cmap=plt.get_cmap('gray'))

# Normalize data
train_X = tf.keras.utils.normalize(train_X, axis=1)
test_X = tf.keras.utils.normalize(test_X, axis=1)
plt.imshow(train_X[10])

plt.imshow(train_X[10], cmap=plt.get_cmap('gray'))

# Resizing image for Convolution
# You always have to give a 4D array as input to the CNN.
train_X = np.array(train_X).reshape(-1, 28, 28, 1)
test_X = np.array(test_X).reshape(-1, 28, 28, 1)
print(train_X.shape)
print(test_X.shape)



"""
# MODEL OPTION 1
# Creating a Deep Neural Network
# Sequential - A feedforward neural network
# Dense - A typical layer in our model
# Dropout - Is used to make the neural network more robust, by reducing overfitting
# Flatten - It is used to flatten the data for use in the dense layer
# Conv2d - We will be using a 2-Dimensional CNN
# MaxPooling2D - Pooling mainly helps in extracting sharp and smooth features. 
#   It is also done to reduce variance and computations. Max-pooling helps in extracting 
#   low-level features like edges, points, etc. While Avg-pooling goes for smooth features.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# Creating the network
model = Sequential()

### First Convolution Layer
# 64 -> number of filters, (3,3) -> size of each kernal,
# For first layer we have to mention the size of input
model.add(Conv2D(64, (3,3), input_shape = (28, 28, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

### Second Convolution Layer
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

### Third Convolution Layer
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

### Fully connected layer 1
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

### Fully connected layer 2
model.add(Dense(32))
model.add(Activation("relu"))

### Fully connected layer 3, output layer must be equal to number of classes
model.add(Dense(10))
model.add(Activation("softmax"))

# SHow info about the model
print(model.summary())
"""



# Compile will indicate the loss function, the optimazer and the metrics
# model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])


"""
# MODEL OPTION 2
# Creating a Deep Neural Network
# Sequential - A feedforward neural network
# Dense - A typical layer in our model
# Dropout - Is used to make the neural network more robust, by reducing overfitting
# Flatten - It is used to flatten the data for use in the dense layer
# Conv2d - We will be using a 2-Dimensional CNN
# MaxPooling2D - Pooling mainly helps in extracting sharp and smooth features. 
#   It is also done to reduce variance and computations. Max-pooling helps in extracting 
#   low-level features like edges, points, etc. While Avg-pooling goes for smooth features.

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))

# compile model
opt = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Show info about the model
print(model.summary())
"""



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_X, train_Y)



# MODEL OPTION 3

from tensorflow.keras.optimizers import SGD
from keras.applications.vgg16 import VGG16

model = VGG16()

# compile model
opt = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Show info about the model
print(model.summary())



# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# print(train_Y)
print(y_train)

# Predict test data
predictions = model.predict([test_X])
print(predictions)

predictions_value = np.argmax(predictions, axis=1)
print(predictions_value)

# Save predictions in file
i=1
with open("predictions.csv", 'w') as f:
    f.write("ImageId,Label\n")
    for value in predictions_value:
        f.write(str(i))
        f.write(',')
        f.write(str(value))
        f.write('\n')
        i=i+1

