import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import math

PATH_TRAINED_WEIGHTS = "training_data/model-weights.ckpt"
checkpoint_callback = keras.callbacks.ModelCheckpoint(PATH_TRAINED_WEIGHTS,
                                                      save_weights_only=True,
                                                      verbose=1)

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# converting the labels to one-hot (representation)
train_labels = keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = keras.utils.to_categorical(test_labels, num_classes=10)

# preprocess the images
train_images = train_images.reshape([-1, 28, 28, 1])
test_images = test_images.reshape([-1, 28, 28, 1])

classes = [str(x) for x in range(10)]

print("Size of:")
print("Training-image: {}".format(train_images.shape))
print("Training-label: {}".format(train_labels.shape))
print("Test-image: {}".format(test_images.shape))
print("Test-label: {}".format(test_labels.shape))


def plot_images(images, labels):
    fig, axes = plt.subplots(nrows=3, ncols=3)
    for i, axes in enumerate(axes.flat):
        axes.imshow(images[i].reshape(28, 28), cmap='binary')
        axes.set_xlabel("Ground Truth : {}".format(classes[np.argmax(labels[i])]))
        axes.set_xticks([])
        axes.set_yticks([])
    plt.show()


plot_images(train_images[:10], train_labels[:10])

# create a Sequential Model using Keras API
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=[28, 28, 1]),
    keras.layers.Conv2D(filters=16, kernel_size=[5, 5], strides=[1, 1], activation=tf.nn.relu),
    keras.layers.MaxPool2D(pool_size=[2, 2]),
    keras.layers.Conv2D(filters=36, kernel_size=[5, 5], strides=[1, 1], activation=tf.nn.relu),
    keras.layers.Flatten(),
    keras.layers.Dense(units=512, activation=tf.nn.relu),
    keras.layers.Dense(units=10, activation=tf.nn.softmax)
])

model.summary()

model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# load previously saved weights into the model
model.load_weights(PATH_TRAINED_WEIGHTS)

# train
"""
model.fit(train_images,
          train_labels,
          epochs=50,
          batch_size=512,
          validation_split=0.1,
          verbose=2,
          callbacks=[checkpoint_callback])
# """

# evaluation accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy : {}'.format(test_acc * 100))


def plot_cnn_weights(weights, input_channel=0):
    """
    Visualizes filters of the given weights
    :param weights: Weight of the convolutional layer, list of len = 4
    :param input_channel: Channel for which the filters are to be visualized
    """
    # Get lowest and highest weights
    weight_min = np.min(weights)
    weight_max = np.max(weights)
    num_filters = weights.shape[3]

    num_rows = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_rows)
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            image = weights[:, :, input_channel, i]
            ax.imshow(
                image,
                vmin=weight_min,
                vmax=weight_max,
                interpolation='nearest',
                cmap='seismic',
            )
            ax.set_xlabel('Filter {}'.format(i))

        ax.set_xticks([])
        ax.set_yticks([])

    # fig.suptitle('Filters for Input Channel {}'.format(input_channel))
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


plot_cnn_weights(model.get_weights()[0], input_channel=0)
plot_cnn_weights(model.get_weights()[2], input_channel=0)
