import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

PATH_TRAINED_WEIGHT = 'training_1/model-weights.ckpt'
cp_callback = keras.callbacks.ModelCheckpoint(PATH_TRAINED_WEIGHT,
                                              save_weights_only=True,
                                              verbose=1)

print('Shape of train_images : {}'.format(train_images.shape))
print('Shape of train_labels : {}'.format(train_labels.shape))


def show_image(i, images):
    """
    Display's image at index @i in the given @images
    """
    plt.figure()
    plt.imshow(images[i])
    plt.colorbar()
    plt.grid(False)
    plt.show()


show_image(0, train_images)

# preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

model = keras.Sequential([
    # transform the format of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dropout(rate=0.15),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

"""
Loss function -> This measures how accurate the model is during training. We want to minimize
                this function to "steer" the model in the right direction.
Optimizer -> This is how the model is updated based on the data it sees and its loss function.
Metrics -> Used to monitor the training and testing steps. The following example uses accuracy,
           the fraction of the images that are correctly classified.
"""

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# training
model.load_weights(PATH_TRAINED_WEIGHT)
model.fit(train_images, train_labels, epochs=20, batch_size=4, callbacks=[cp_callback])

# evaluating accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)


# visualizing
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def visualize(x, prediction):
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(x, prediction, test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(x, prediction, test_labels)


predictions = model.predict(test_images)
visualize(0, predictions)
