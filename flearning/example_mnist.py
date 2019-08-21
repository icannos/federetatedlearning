

from agents import Agent, MasterAgent
import tensorflow as tf
from keras.layers import Dense
from keras.objectives import categorical_crossentropy
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt


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

# We load the mnist data set
mnist =  tf.keras.datasets.mnist

# some preprocessing, and get training set and validation set
(x_train, y_train),(x_test, y_test) = mnist.load_data()
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

agents = [Agent(session, f"agent{i}", input_shape, output_shape,
                batch_size=128, f_mk_model=mk_model, data=datasets[i]) for i in range(nb_agent)]

master = MasterAgent(session, f"master", input_shape, output_shape, f_mk_model=mk_model, agents=agents,
                     data=(x_test, y_test_cat))

# We gets the history: loss and metrics for each training step
loss_history, metrics_history = master.train(100)

metrics_history = np.array(metrics_history)

X = [i for i in range(len(loss_history))]

# For demonstration we display it
plt.plot(X, loss_history)
plt.plot(X, metrics_history[:, 0])

plt.savefig("img/training_graph.svg", format="svg")


#Example json conversion/ipfs manipulation
from utils import weights2json, jsontoweights
import ipfshttpclient
import json

client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001/http')
i = 0
res = []
for a in agents:
    json_weights = weights2json(a.get_weights())
    with open('weights'+str(i)+'.json', 'w', encoding='utf-8') as outfile:
        json.dump(json_weights, outfile, ensure_ascii=False, indent=2)

    res.append(client.add('weights'+str(i)+'.json'))
    print(res[i]['Hash'])
    client.get(res[i]['Hash'])

    with open(res[i]['Hash']) as json_file:
        ipfs_weights = json.load(json_file)
        print(jsontoweights(ipfs_weights))
    i += 1
    # print(json_weights)
    # hash = client.add_json(json_weights)
    # print(hash)
    # client.cat(hash)
    # from_IPFS_weights = client.get_json(hash)
    # print(from_IPFS_weights)
    # print(jsontoweights(from_IPFS_weights))
