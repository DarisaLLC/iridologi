import random
import pandas as pd
import numpy as np
from random import seed
from math import exp
from sklearn.metrics import confusion_matrix

# Initialize Network
def initialize_network(n_inputs, n_hidden, n_layers, n_outputs):
    network = list()
    seed(4)
#     for i in range (n_layers):
#         hidden_layer = [{'weights': [round(random.uniform(0,0.5),2) for i in range(n_inputs+1)]} for i in range(n_hidden)]
#         if i > 0:
#             hidden_layer = [{'weights': [round(random.uniform(0,0.5),2) for i in range(n_hidden+1)]} for i in range(n_hidden)]
#         network.append(hidden_layer)
#     output_layer = [{'weights': [round(random.uniform(0,0.5),2) for i in range(n_hidden+1)]} for i in range(n_outputs)]
    for i in range (n_layers):
        hidden_layer = [{'weights': [round(random.random(),2) for i in range(n_inputs+1)]} for i in range(n_hidden)]
        if i > 0:
            hidden_layer = [{'weights': [round(random.random(),2) for i in range(n_hidden+1)]} for i in range(n_hidden)]
        network.append(hidden_layer)
    output_layer = [{'weights': [round(random.random(),2) for i in range(n_hidden+1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

# FORWARD PROPAGATE
# 1. Neuron Activation
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# 2. Neuron Transfer
def transfer(activation):
    return 1.0 / (1.0+exp(-activation))

# 3. Forward Propagation
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# BACK PROPAGATE
# 1. Transfer Derivative
def transfer_derivative(output):
    return output * (1.0 - output)

# 2. Error Backpropagation
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i !=len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# 3. Update Weights
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']

# TRAIN NETWORK
def train_network(network, train, l_rate, loss_limit, n_outputs):
    epoch = 0
    while True:
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        # print('>epoch=%d, l_rate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
        epoch+=1
        if epoch == 20000:
            break
        if sum_error <= loss_limit:
            break
    print('>epoch=%d, l_rate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

# MAKING PREDICTION
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

# CHECK ACCURACY
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct/float(len(actual)) * 100.0


# BACKPROPAGATION HANDLER
def back_propagation(train, test, l_rate, loss_limit, n_layers, n_hidden, name):
    n_inputs = len(train[0])-1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_layers, n_outputs)
    train_network(network, train, l_rate, loss_limit, n_outputs)
    pd.DataFrame(np.array(network)).to_csv(name, header=False, index=False)

    test_set = pd.DataFrame(test).iloc[:, :-1]
    test_set = np.array(test_set)

    predictions = list()
    for row in test_set:
        prediction = predict(network, row)
        predictions.append(prediction)
    print('PREDICTIONS:')
    print(predictions)

    expected = [int(row[-1]) for row in test]
    print('EXPECTED:')
    print(expected)

    accuracy = accuracy_metric(expected, predictions)
    print('Accuracy = %.3f%%' %(accuracy))
    print('\nConfusion Matrix:')

    print(confusion_matrix(expected, predictions))
    tn, fp, fn, tp = confusion_matrix(expected, predictions).ravel()
    print('TN={}, FP={}, FN={}, TP={}'.format(tn, fp, fn, tp))
    return network
