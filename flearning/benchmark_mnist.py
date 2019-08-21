

from agents import Agent, MasterAgent
from benchmarks_agents import benchmark_agent
import tensorflow as tf
from keras.layers import Dense
from keras.objectives import categorical_crossentropy
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
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
        h1 = tf.keras.layers.Dense(128, activation="relu")(x)
        h2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(h1)
        preds = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(h2)

        loss = tf.reduce_mean(categorical_crossentropy(y, preds))

        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return preds, loss, [accuracy]


# ============================================= #

np.random.seed(6547892)

optimizer = ""
steps_by_step = 16
batch_size = 16
epochs = 500
dataset_size = 600
fusion_mode = "mean"

# We load the mnist data set
mnist =  tf.keras.datasets.mnist


# some preprocessing, and get training set and validation set
(x_train, y_train),(x_test, y_test) = mnist.load_data()

n = x_train.shape[0]
idx = np.arange(n)
np.random.shuffle(idx)
x_train, y_train = x_train[0:dataset_size], y_train[0:dataset_size]

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = np.reshape(x_train, (x_train.shape[0], 784))
x_test = np.reshape(x_test, (x_test.shape[0], 784))

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# How many distributed agents we want to use
nb_agent = 3

data_size = x_train.shape[0]
m = int(data_size / nb_agent)

input_shape = (784,)
output_shape = (None,)

# We build the training set owned by each agent
datasets = [(x_train[m*i:m*(i+1)], y_train_cat[m*i:m*(i+1)]) for i in range(nb_agent)]

# Open a tensorflow session
session = tf.Session()

# We construct the list of agents, then we pass it to the master
# I will come back on that point in the documentation. Since here it is a simulation we can pass it the true agent
# object. However in production, we will need to write a communication class, which will handle the communication with
# the different nodes. (For now it will be tackle by the DTP system)

agents = [benchmark_agent(session, f"agent{i}", input_shape, output_shape,
                batch_size=batch_size, f_mk_model=mk_model, data=datasets[i], validation_set=(x_test, y_test_cat), steps_by_step=steps_by_step) for i in range(nb_agent)]

master = MasterAgent(session, f"master", input_shape, output_shape, f_mk_model=mk_model, agents=agents,
                     data=(x_test, y_test_cat), fusion_mode=fusion_mode)

# We gets the history: loss and metrics for each training step
loss_history, metrics_history = master.train(epochs)

metrics_history = np.array(metrics_history)

X = [i for i in range(len(loss_history))]

d = pd.DataFrame()
d["collab_loss"] = pd.Series(loss_history)
d["collab_accuracy"] = pd.Series(metrics_history[:, 0])

# For demonstration we display it
plt.plot(X, loss_history, label="Collaborative learning -- Loss")
plt.plot(X, metrics_history[:, 0], label="Collaborative learnig -- Accuracy")

for i,a in enumerate(master.agents):
    metrics = np.array(a.local_metrics_history)

    plt.plot(X, a.local_loss_history, label=f"Agent-{i} -- Loss")
    plt.plot(X, metrics[:, 0], label=f"Agent-{i} -- Accuracy")

    d[f"agent{i}_loss"] = pd.Series( a.local_loss_history)
    d[f"agen{i}_accuracy"] = pd.Series(metrics[:, 0])

plt.legend(loc='upper right')


plt.savefig(f"img/{optimizer}-{fusion_mode}_mnist_collab_vs_alone-{dataset_size}-{epochs}-{batch_size}-{steps_by_step}.svg", format="svg")
d.to_csv(f"benchmarks/{optimizer}-{fusion_mode}_benchmark_mnist-{dataset_size}-{epochs}-{batch_size}-{steps_by_step}.csv")

# Change the path to where you want, it will dump the model at the end in order to allow us to make test on it
master.save("models/model1")



