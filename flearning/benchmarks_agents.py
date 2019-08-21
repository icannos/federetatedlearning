import tensorflow as tf
from agents import Agent, MasterAgent
import numpy as np


class benchmark_agent(Agent):
    def __init__(self, session, scope, input_shape, output_shape, f_mk_model, data, batch_size, steps_by_step=20,
                 validation_set=None, test_batch_size=10 ** 6):
        super().__init__(session, scope, input_shape, output_shape, f_mk_model, data, batch_size, steps_by_step)

        self.test_batch_size = test_batch_size
        self.validation_set = validation_set
        with tf.variable_scope(self.scope):
            self.local_model_output, self.local_loss, self.local_metrics = \
                self.mk_model(scope="local", x=self.input_ph, y=self.output_ph)

        self.local_model_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope + "/local")

        self.local_train = self.mk_local_train_op()

        self.init_op = tf.group(tf.global_variables_initializer(),
                                tf.local_variables_initializer())
        self.session.run([self.init_op])

        self.local_loss_history = []
        self.local_metrics_history = []

    def mk_local_train_op(self):
        """
        Build the tensorflow training op.
        :return: tensorflow op to perform a gradient descent step.
        """
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(self.local_loss)

        return train_op

    def training_step(self):
        """
        Perform the training for one agent. Sample a batch of data and use it to make a training step.
        :return:
        """
        dataX = self.data[0]
        dataY = self.data[1]

        for step in range(self.steps_by_step):
            if self.shuffle:
                idx = np.random.randint(0, dataX.shape[0], self.batch_size)
            else:
                begin = np.random.randint(0, dataX.shape[0] - self.batch_size, 1)
                end = begin + self.batch_size
                idx = np.arange(begin, end)

            X = dataX[idx]
            Y = dataY[idx]

            self.session.run([self.train_op, self.local_train],
                             feed_dict={self.input_ph: X, self.output_ph: Y})

        valX = self.validation_set[0]
        valY = self.validation_set[1]

        if valX.shape[0] > self.test_batch_size:
            ids = np.array_split(np.arange(valX.shape[0]), valX.shape[0] // self.test_batch_size)
        else:
            ids = [np.arange(valX.shape[0])]

        local_loss = []
        metrics_history = []
        
        for index in ids:
            loc_loss = self.session.run(self.local_loss,
                                        feed_dict={self.input_ph: valX[index], self.output_ph: valY[index]})

            local_loss.append(loc_loss)

            metrics = self.session.run(self.local_metrics,
                                       feed_dict={self.input_ph: valX[index], self.output_ph: valY[index]})

            metrics_history.append(metrics)

        local_loss = np.mean(local_loss)
        metrics_history = np.mean(metrics_history, axis=0)


        self.local_loss_history.append(local_loss)
        self.local_metrics_history.append(metrics_history)

    def eval_loss(self, X, Y):
        valX = X
        valY = Y

        if valX.shape[0] > self.test_batch_size:
            ids = np.array_split(np.arange(valX.shape[0]), valX.shape[0] // self.test_batch_size)
        else:
            ids = [np.arange(valX.shape[0])]

        local_loss = []

        for index in ids:
            loc_loss = self.session.run(self.loss,
                                        feed_dict={self.input_ph: valX[index], self.output_ph: valY[index]})

            local_loss.append(loc_loss)

        local_loss = np.mean(local_loss)

        return local_loss

    def eval_metrics(self, X, Y):
        valX = X
        valY = Y

        if valX.shape[0] > self.test_batch_size:
            ids = np.array_split(np.arange(valX.shape[0]), valX.shape[0] // self.test_batch_size)
        else:
            ids = [np.arange(valX.shape[0])]

        metrics_history = []

        for index in ids:

            metrics = self.session.run(self.metrics,
                                       feed_dict={self.input_ph: valX[index], self.output_ph: valY[index]})

            metrics_history.append(metrics)

        metrics_history = np.mean(metrics_history, axis=0)

        return metrics_history
        
        
class benchmark_MasterAgent(benchmark_agent):
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
        self.saver.save(self.session, path)

    def load(self, path):
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

