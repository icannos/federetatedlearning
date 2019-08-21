

from agents import Agent, MasterAgent
import tensorflow as tf
from keras.objectives import categorical_crossentropy
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dropout, MaxPooling2D, Dense, Flatten
import pandas as pd

def mk_model(scope, x, y):
    """
    Implement the model to use for this use case. (I have made the flearning lib use case agnostic
    :param scope: Tensorflow namespace to use
    :param x: input tensor
    :param y: output tensor
    :return: preds operation, loss operation, init operation, list of opperation (different metrics that can be computed
    to test the training).
    """

    with tf.variable_scope(scope):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=x_train.shape[1:]))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Activation('softmax'))

        preds = model(x)

        loss = tf.reduce_mean(categorical_crossentropy(y, preds))

        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init_op = tf.global_variables_initializer()

    return preds, loss, init_op, [accuracy]


# ============================================= #

# We load the mnist data set
mnist =  tf.keras.datasets.cifar10

# some preprocessing, and get training set and validation set
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# How many distributed agents we want to use
nb_agent = 3

data_size = x_train.shape[0]
m = data_size // nb_agent

input_shape = x_train.shape[1:]
output_shape = (10, )

# We build the training set owned by each agent
datasets = [(x_train[m*i:m*(i+1)], y_train_cat[m*i:m*(i+1)]) for i in range(nb_agent)]

# Open a tensorflow session
session = tf.Session()

# We construct the list of agents, then we pass it to the master
# I will come back on that point in the documentation. Since here it is a simulation we can pass it the true agent
# object. However in production, we will need to write a communication class, which will handle the communication with
# the different nodes. (For now it will be tackle by the DTP system)

agents = [Agent(session, f"agent{i}", input_shape, output_shape,
                batch_size=512, f_mk_model=mk_model,data=datasets[i]) for i in range(nb_agent)]

master = MasterAgent(session, f"master", input_shape, output_shape, f_mk_model=mk_model, agents=agents,
                     data=(x_test, y_test_cat))

# We gets the history: loss and metrics for each training step
loss_history, metrics_history = master.train(100)

metrics_history = np.array(metrics_history)

X = [i for i in range(len(loss_history))]



# # For demonstration we display it
# plt.plot(X, loss_history)
# plt.plot(X, metrics_history[:, 0])
#
# plt.savefig("img/training_graph_ciphar.svg", format="svg")


