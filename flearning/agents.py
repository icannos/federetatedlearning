import tensorflow as tf
import numpy as np
import random
from keras.utils import to_categorical


class Agent:
    """
    Implement the basic agent. This is used by every slave and it is the basis of the master.
    This embedded training and datamanagement for one particular agent.
    """

    def __init__(self, session, scope, input_shape, output_shape, f_mk_model, data, batch_size, steps_by_step=10,
                 shuffle=False, logs=True):
        """

        :param session: Tensorflow session make the computations
        :param scope: Namespace for this agent, a uniqid
        :param input_shape: input shape of the model
        :param output_shape: output shape
        :param f_mk_model: Function that build tensorflow op for the model. Should return:
        output_op, loss_op, init_op for the model
        :param data: The dataset for this agent. (x_train, y_train)
        :param batch_size: Size of a training batch
        :param steps_by_step: number of training step to proceed on this client
        """

        self.logs = logs
        self.shuffle = shuffle
        self.steps_by_step = steps_by_step
        self.batch_size = batch_size
        self.data = data
        self.session = session
        self.scope = scope
        self.mk_model = f_mk_model

        with tf.variable_scope(self.scope):
            self.input_ph = tf.placeholder("float32", shape=(None, *input_shape))
            self.output_ph = tf.placeholder("float32", shape=(None, *output_shape))

            self.model_output, self.loss, self.metrics = \
                self.mk_model(scope="model", x=self.input_ph, y=self.output_ph)

            self.model_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + "/model")

            self.weights_ph = [tf.placeholder("float32", shape=w.shape) for w in self.model_weights]
            self.weights_shape = [w.shape for w in self.model_weights]

            self.train_op = self.mk_train_op()

            self.set_weights_op = [tf.assign(mw, wph) for mw, wph in zip(self.model_weights, self.weights_ph)]

            self.init_op = tf.group(tf.global_variables_initializer(),
                                    tf.local_variables_initializer())  # tf.global_variables_initializer()
            self.session.run([self.init_op])


    def set_weights(self, weights):
        """
        Copy the weights passed as parameters into the model variables. It is used to put the weights received from
        the master.
        :param weights:
        :return:
        """
        feed_dict = {wph: w for wph, w in zip(self.weights_ph, weights)}
        self.session.run(self.set_weights_op, feed_dict=feed_dict)

    def get_weights(self):
        """
        Get the weights of the model. it used after a training step to share the updates with the master.
        :return: weights of the model
        """
        return self.session.run(self.model_weights)

    def mk_train_op(self):
        """
        Build the tensorflow training op.
        :return: tensorflow op to perform a gradient descent step.
        """
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)

        train_op = optimizer.minimize(self.loss)

        return train_op

    def training_step(self):
        """
        Perform the training for one agent. Sample a batch of data and use it to make a training step.
        :return:
        """
        dataX = self.data[0]
        dataY = self.data[1]

        for _ in range(self.steps_by_step):
            if self.shuffle:
                idx = np.random.randint(0, dataX.shape[0], self.batch_size)
            else:
                begin = np.random.randint(0, dataX.shape[0] - self.batch_size, 1)
                end = begin + self.batch_size
                idx = np.arange(begin, end)

            X = dataX[idx]
            Y = dataY[idx]

            self.session.run(self.train_op, feed_dict={self.input_ph: X, self.output_ph: Y})

    def predict(self, X):
        """
        Returns the predictions for X
        :param X: inputs data
        :return: predictions
        """
        return self.session.run([self.model_output], feed_dict={self.input_ph: X})[0]

    def eval_loss(self, X, Y):
        """
        Computes the loss on the presented data set
        :param X: inputs
        :param Y: expected output
        :return: loss on this dataset
        """
        return self.session.run([self.loss], feed_dict={self.input_ph: X, self.output_ph: Y})[0]

    def eval_metrics(self, X, Y):
        """
        Computes the different metrics returned by the model
        :param X: inputs
        :param Y: expected output
        :return: loss on this dataset
        """
        if self.metrics is not None:
            return self.session.run(self.metrics, feed_dict={self.input_ph: X, self.output_ph: Y})
        else:
            return None


class MasterAgent(Agent):
    """
    Represent the master agent. This is an agent, therefore it also can train a part of the model using its own data.
    It handles the communications between the different agents of the network.

    Communication methods
    ---------------------
    broadcast_weights unit --> unit should be use to send weights to every agent (it sends the master's weight)
    retrieve_weights unit --> list of list of weights (weights[i][j] is the j-th weight of agent i )
    retrieve the weights from every agent.

    """

    def __init__(self, session, scope, input_shape, output_shape, f_mk_model, agents, data, fusion_mode="mean"):
        """

        :param session: Session used for tensorflow computations
        :param scope: Namespace for this agent
        :param input_shape: input shape of the model
        :param output_shape: output shape of the model
        :param f_mk_model:
        :param agents: List of Agent-like object representing each slave of the network. This can be used to implement
        true communications. One could implement a communication Agent class, which should supports set_weights and
        get_weights. This class could handle the communication with the real agent object instantiated on the slave
        server.
        :param data: Useless if master does not proceed to training
        """
        super().__init__(session, scope, input_shape, output_shape, f_mk_model, batch_size=0, data=[])

        self.fusion_mode = fusion_mode
        self.agents = agents
        self.data = data

        self.saver = tf.train.Saver()

    def save(self, path):
        """
        This dumps the model to path (see tensorflow save and restore for the dump format)
        :param path: path where to save the model
        :return: None
        """
        self.saver.save(self.session, path)

    def load(self, path):
        """
        Restore the variables of the model
        :param path: path to the model (see tensof
        :return:
        """
        self.saver.restore(self.session, path)

    def train(self, episodes):
        """
        Training function. It commands to each to make a training step, retrieve the new weights, combine them and
        rebroadcast them to each agent.
        :param episodes: Number of training epoch to make
        :return: 2 lists containing the loss history and the differents metrics
        """

        loss_history = []
        metrics_history = []

        for e in range(episodes):
            print(f"epoch: {e}")
            for a in self.agents:
                a.training_step()

            weights = self.retrieve_weights()

            new_weights = self.combine_weights(weights)

            self.set_weights(new_weights)
            self.broadcast_weights()

            # Just for testing returns the loss

            if self.logs:
                loss_history.append(self.eval_loss(self.data[0], self.data[1]))
                metrics_history.append(self.eval_metrics(self.data[0], self.data[1]))

        return loss_history, metrics_history

    def combine_weights(self, weights):
        """
        Averages the weights of the different slave into the new master parameters.
        :param weights: list of numpy array corresponding to the weights of the model
        :return: new weights from combition of each slave
        """

        if self.fusion_mode == "mean":
            return self.mean_fusion(weights)
        elif self.fusion_mode == "grad":
            return self.grad_fusion(weights)

    def grad_fusion(self, weights):
        W0 = self.get_weights()
        new_weights = [w for w in W0]

        for w in weights:
            for i in range(len(w)):
                new_weights[i] += w[i] - W0[i]

        return new_weights

    def mean_fusion(self, weights):
        new_weights = [np.zeros(s) for s in self.weights_shape]

        for w in weights:
            for i in range(len(w)):
                new_weights[i] += w[i] / len(weights)

        return new_weights

    def broadcast_weights(self):
        """
        Send master's weights to each slave.
        :return: None
        """
        for a in self.agents:
            a.set_weights(self.get_weights())

    def retrieve_weights(self):
        """
        Get the weights of each slave and returns them
        :return: List of list of weights (weights[i][j] is the j-th weight of agent i)
        """
        weights = []

        for a in self.agents:
            weights.append(a.get_weights())

        return weights
