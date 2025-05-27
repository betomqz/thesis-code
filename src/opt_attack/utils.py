import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import keras
import tensorflow as tf
import os

class TextColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'  # End of color


def load_mnist_data():
    '''
    Function to load an process MNIST data. Code is not mine.
    TODO: paste link from keras documentation.
    '''

    # Model / data parameters
    num_classes = 10

    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    # print("x_train shape:", x_train.shape)
    # print(x_train.shape[0], "train samples")
    # print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test

def get_inputs_tuples():
    '''
    Returns the inputs in ordered tuples corresponding to each digit
    '''
    x_train, x_test, y_train, y_test = load_mnist_data()
    # Choose inputs
    inputs = [
        (0, x_test[3]),
        (1, x_test[2]),
        (2, x_test[1]),
        (3, x_test[18]),
        (4, x_test[4]),
        (5, x_test[8]),
        (6, x_test[11]),
        (7, x_test[0]),
        (8, x_test[61]),
        (9, x_test[7])
    ]
    return inputs


def train_nn_mnist(save_path='models/', with_softmax=False):

    batch_size = 128
    epochs = 3
    num_classes = 10
    input_shape = (28, 28, 1)

    x_train, x_test, y_train, y_test = load_mnist_data()

    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes)
        ]
    )
    if with_softmax:
        model.add(keras.layers.Activation("softmax"))

    # Idea from https://github.com/carlini/nn_robust_attacks/blob/master/train_models.py
    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)

    loss_fn = "categorical_crossentropy" if with_softmax else fn

    model.compile(
        loss=loss_fn, optimizer="adam", metrics=["accuracy"]
    )

    model.fit(
        x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1
    )

    # Export and save model
    name = "softmaxmnist.keras" if with_softmax else "mnist.keras"
    model.save(save_path + name)
    return model


def load_mnist_model(path='models/mnist.keras'):
    '''
    Function to load pre-trained MNIST model.

    Returns pre-trained model located on `path`.

    **Note for self**: `path` on Windows is only 'models/mnist.keras'
    '''
    num_classes = 10
    input_shape = (28, 28, 1)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes)
        ]
    )
    model.load_weights(path)
    return model


def vis_flat_mnist(x, save=False, filename="fig.png", format="png"):
    '''
    Function to visualize an array of size 784 as a 28x28
    image in black and white.
    '''
    temp = x.reshape(28,28,1)
    fig, ax = plt.subplots()
    ax.set_axis_off()
    plt.imshow(temp, cmap='gray_r')
    if save:
        plt.savefig(filename, format=format, pad_inches=0, bbox_inches='tight')
    plt.show()

def save_flat_mnist_fig(x, fname):
    temp = x.reshape(28,28,1)
    fig, ax = plt.subplots()
    ax.set_axis_off()
    plt.imshow(temp, cmap='gray_r')
    plt.savefig(fname, pad_inches=0, bbox_inches='tight')
    plt.close(fig)

def eval_flat_pred(x, model):
    if x is None:
        return -1

    temp = x.reshape(28,28,1)
    return np.argmax(model.predict(temp[np.newaxis,...],verbose = 0))


def big_graph(path):
    '''
    Function to generate a "big graph" of all available images
    in a given path.
    '''
    image_size = (28, 28)
    rows, cols = 10, 10

    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    for i in range(10):
        for j in range(10):
            file_name = f"{path}/{i}-to-{j}.png"
            if os.path.exists(file_name):
                img = mpimg.imread(file_name)
            else:
                img = np.array(Image.new('L', image_size, 'white'))

            axes[i, j].imshow(img, cmap='gray')
            # axes[i, j].axis('off')
            axes[i, j].set_xticklabels([])
            axes[i, j].set_yticklabels([])
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    # Add row and column labels
    for ax, col in zip(axes[9], range(cols)):
        ax.set_xlabel(f'{col}')

    for ax, row in zip(axes[:,0], range(rows)):
        ax.set_ylabel(f'{row}')

    # Add axis titles
    fig.text(0.5, 0.04, 'Target', ha='center', va='center', fontsize='large')
    fig.text(0.04, 0.5, 'Source', ha='center', va='center', rotation='vertical', fontsize='large')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(rect=[0.05, 0.05, 1, 1])

    # Save the combined image
    plt.savefig(f"{path}/all.png")
    plt.show()
