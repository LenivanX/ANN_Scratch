import numpy as np
import random


class ANN:
    def __init__(self, layout):
        # NN layout
        self.layout = np.array(layout)

        # dictionary of cost function(s)
        self.costs = {
            'mse': self.mse,
            'binary_cross_entropy': self.binary_cross_entropy
        }

        # dictionary of activation functions
        self.activations = {
            'sigmoid': self.sigmoid
        }

        # build the neural network
        self.build()

    # Build the NN by initialising weights and biases
    def build(self):
        # dictionaries of weights and biases
        self.weights = {}
        self.biases = {}

        # fix random seed to 0
        np.random.seed(0)

        # initialising weights and biases
        for layer in range(1, self.layout.size):
            self.weights['w' + str(layer)] = np.random.uniform(-0.5, 0.5, (self.layout[layer], self.layout[layer - 1]))
            self.biases['b' + str(layer)] = np.random.uniform(-0.5, 0.5, (self.layout[layer], 1))

    # Mean Squared Error
    def mse(self, a, y, deriv=False):
        if deriv:
            return a - y
        return (1 / a.size) * np.sum((a - y) ** 2)

    # Binary Cross Entropy
    def binary_cross_entropy(self, a, y, deriv=False):
        if deriv:
            return (1 / len(a)) * (a - y)
        epsilon = 1e-15
        a_2 = [max(i, epsilon) for i in a]
        a_2 = [min(i, 1 - epsilon) for i in a]
        a = np.array(a_2)
        return -np.mean(y * np.log(a) + (1 - y) * np.log(1 - a))

    # Sigmoid Activation function
    def sigmoid(self, z, deriv=False):
        if deriv:
            return self.sigmoid(z) * (1 - self.sigmoid(z))
        return 1 / (1 + np.exp(-z))

    # Forward Propagation
    def forward(self, x, activation):
        a = [x]
        z = [x]

        for layer in range(1, self.layout.size):
            z_i = self.weights['w' + str(layer)] @ np.array(a[-1]) + self.biases['b' + str(layer)]
            if layer == self.layout.size - 1:
                a_i = self.activations[activation](z_i)
            else:
                a_i = self.sigmoid(z_i)
            z.append(z_i)
            a.append(a_i)
        return a, z

    # Backward Propagation
    def backward(self, a, z, activation, cost, y, alpha):
        delta = self.costs[cost](a[-1], y, deriv=True)

        for layer in range(self.layout.size - 1, 0, -1):
            self.weights['w' + str(layer)] += -alpha * delta @ a[layer - 1].T
            self.biases['b' + str(layer)] += -alpha * delta
            delta = self.activations[activation](z[layer - 1], deriv=True) * self.weights['w' + str(layer)].T @ delta

    def fit(self, X, Y, activation='sigmoid', cost='mse', alpha=0.05, epochs=100):
        # convert inputs into numpy arrays
        X = np.array(X)
        Y = np.array(Y)
        try:
            X.shape[1]
        except IndexError:
            X = X.reshape(len(X), 1)
        try:
            Y.shape[1]
        except IndexError:
            Y = Y.reshape(len(Y), 1)

        for epoch in range(epochs):
            # random index for Stochastic Gradient Descent
            random_index = random.randint(0, len(X) - 1)

            # inputs at the random index
            x = X[random_index]
            y = Y[random_index]
            x.shape += (1,)
            y.shape += (1,)

            # forward propagation function returning activations of each layer
            a, z = self.forward(x, activation)

            # cost of current epoch
            C = self.costs[cost](a[-1], y)
            self.cost = C

            # backward propagation using activationd of each layer
            self.backward(a, z, activation, cost, y, alpha)

    def predict(self, X):
        X = np.array(X)
        try:
            X.shape[1]
        except IndexError:
            X = X.reshape(len(X), 1)

        predictions = []
        for x in X:
            x.shape += (1,)
            a = x
            for layer in range(1, self.layout.size):
                z = self.weights['w' + str(layer)] @ a + self.biases['b' + str(layer)]
                a = self.sigmoid(z)
            if a.shape[0] == 1:
                predictions.append(a[0][0])
            else:
                predictions.append(a.T[0])
        return np.vstack(predictions)

    def score(self, X, Y):
        Y = np.array(Y)
        A = np.array(self.predict(X))

        flag = False
        try:
            A.shape[1]
        except IndexError:
            flag = True

        correct = 0
        total = 0
        for a, y in zip(A, Y):
            if flag:
                if (a >= 0.5 and y == 1) or (a < 0.5 and y == 0):
                    correct += 1
            else:
                if np.argmax(a.T) == np.argmax(y):
                    correct += 1
            total += 1
        return correct / total
