import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

PATH_TRAINED_WEIGHT = 'training_2/model-weights.ckpt'
cp_callback = keras.callbacks.ModelCheckpoint(PATH_TRAINED_WEIGHT,
                                              save_weights_only=True,
                                              verbose=1)

print('Shape of train_images : {}'.format(train_images.shape))
print('Shape of train_labels : {}'.format(train_labels.shape))

# pre-processing the data by normalizing it
train_images = train_images.reshape([-1, 28, 28, 1])
test_images = test_images.reshape([-1, 28, 28, 1])
train_images = train_images / 255.0
test_images = test_images / 255.0

# each label is not a 1-d vector of size 10, but a value
# corresponding to the correct class. We therefore need to convert
# the current representation of the labels to “One Hot Representation”.
train_labels = keras.utils.to_categorical(y=train_labels, num_classes=10)
test_labels = keras.utils.to_categorical(y=test_labels, num_classes=10)

model = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', input_shape=(28, 28, 1)),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(units=1024, activation=tf.nn.relu),
    keras.layers.Dropout(rate=0.3),
    keras.layers.Dense(units=10, activation=tf.nn.softmax)
])

model.summary()

model.compile(
    keras.optimizers.Adam(lr=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.load_weights(PATH_TRAINED_WEIGHT)
"""model.fit(
    x=train_images,
    y=train_labels,
    validation_split=0.10,
    batch_size=16,
    epochs=25,
    verbose=1,
    callbacks=[cp_callback]
)
"""
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
