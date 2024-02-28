import numpy as np


def weight_layer(n_prior_neurons, n_after_neurons):  # returns weight array
    bias_matrix = np.zeros(n_after_neurons)  # todo add bias later
    return np.random.rand(n_prior_neurons, n_after_neurons)


def input_activation(n_input_neurons):  # returns input array
    return np.random.rand(n_input_neurons)


def activation(matrix1, matrix2):  # returns relu(W*X + B)
    _ = np.dot(matrix1, matrix2)
    for i, j in enumerate(_):
        _[i] = max(j, 0)
    return _


def forward_prop():
    a = input_activation(10)
    a1 = activation(a, weight_layer(10, 4))
    a2 = activation(a1, weight_layer(4, 2))
    return a2


def back_prop():  # todo back prop
    pass


print(forward_prop())
