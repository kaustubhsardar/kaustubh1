#python  version 1.3
import numpy as np
from load_mnist import mnist
import matplotlib.pyplot as plt
import pdb
import sys, ast

def tanh(Z):
    A = np.tanh(Z)
    cache = {}
    cache["Z"] = Z
    return A, cache


def tanh_der(dA, cache):
    #A = cache["Z"]
    #A1 = 1 - np.square(np.tanh(A))
    #dZ = 1 - A1
    dZ = dA * (1.0 - (np.tanh(cache["Z"])) ** 2)
    return dZ


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = {}
    cache["Z"] = Z
    return A, cache


def sigmoid_der(dA, cache):
    A = cache["Z"]
    A1, t = sigmoid(A)
    A2 = (1 - A1) * (A1)
    dZ = dA * A2
    return dZ


def initialize_2layer_weights(n_in, n_h, n_fin):
    np.random.seed(0)
    W1 = np.random.randn(n_h, n_in) * 0.01  # .random.randn
    b1 = np.random.randn(n_h, 1) * 0.01
    W2 = np.random.randn(1, n_h) * 0.01
    b2 = np.random.randn(1, 1) * 0.01

    parameters = {}
    parameters["W1"] = W1
    parameters["b1"] = b1
    parameters["W2"] = W2
    parameters["b2"] = b2

    return parameters


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = {}
    cache["A"] = A
    return Z, cache


def layer_forward(A_prev, W, b, activation):
    Z, lin_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, act_cache = sigmoid(Z)
    elif activation == "tanh":
        A, act_cache = tanh(Z)

    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache

    return A, cache


def cost_estimate(A2, Y):
    #epsilon = 1e-14
    #cost = -(Y.dot(np.log(A2 + epsilon).T) + (1 - Y).dot(np.log(1 - A2 + epsilon).T))[0][0]/Y.shape[1]
    epsilon = 1e-5
    m = Y.shape[1]
    lp = np.multiply(np.log10(A2 + epsilon), Y) + np.multiply((1 - Y), np.log10(1 - A2 + epsilon))
    cost = -np.sum(lp) / m
    return cost


def linear_backward(dZ, cache, W, b):
    A = cache["A"]
    dW = np.dot(dZ, A.T)
    db = np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def layer_backward(dA, cache, W, b, activation):
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]

    if activation == "sigmoid":
        dZ = sigmoid_der(dA, act_cache)
    elif activation == "tanh":
        dZ = tanh_der(dA, act_cache)
    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db


def classify(X, parameters):
    activation = "sigmoid"
    act = "tanh"
    A1, t1 = layer_forward(X, parameters["W1"], parameters["b1"], act)
    A2, t2 = layer_forward(A1, parameters["W2"], parameters["b2"], activation)
    YPred = []
    for a in np.nditer(A2):
        if a >= 0.5:
            YPred.append(1)
        else:
            YPred.append(0)

    YPred = np.matrix(YPred)

    return YPred


def two_layer_network(X, Y, X1,Y1, net_dims, num_iterations=2000, learning_rate=0.1):
    n_in, n_h, n_fin = net_dims
    parameters = initialize_2layer_weights(n_in, n_h, n_fin)
    activation = "sigmoid"
    act="tanh"
    A0 = X
    Z0=X1
    costs = []
    costs1= []
    for ii in range(num_iterations):
        # Forward propagation
        ### CODE HERE
        A1, cache1 = layer_forward(A0, parameters["W1"], parameters["b1"], act)
        A2, cache2 = layer_forward(A1, parameters["W2"], parameters["b2"], activation)

        Z1, c1 = layer_forward(Z0, parameters["W1"], parameters["b1"], act)
        Z2, c2 = layer_forward(Z1, parameters["W2"], parameters["b2"], activation)

        # cost estimation
        ### CODE HERE
        cost = cost_estimate(A2, Y)
        cost1 = cost_estimate(Z2, Y1)
        # Backward Propagation
        ### CODE HERE

        epsilon = 0.0001
        m = Y.shape[1]
        dA = 1.0 / m * (np.divide(-1 * Y, A2 + epsilon) + np.divide(1 - Y, 1 - A2 + epsilon))

        dA_prev2, dW2, db2 = layer_backward(dA, cache2, parameters["W2"], parameters["b2"], activation)
        dA_prev1, dW1, db1 = layer_backward(dA_prev2, cache1, parameters["W1"], parameters["b2"], act)
        # update parameters
        ### CODE HERE
        parameters["W1"] = parameters["W1"] - learning_rate * dW1
        parameters["b1"] = parameters["b1"] - learning_rate * db1
        parameters["W2"] = parameters["W2"] - learning_rate * dW2
        parameters["b2"] = parameters["b2"] - learning_rate * db2

        if ii % 10 == 0:
            costs.append(cost)
            costs1.append(cost1)
        if ii % 100 == 0:
            print("Cost at iteration %i is: %f" % (ii, cost))

    return costs, parameters,costs1


def main():
    # getting the subset dataset from MNIST

    net_dims = ast.literal_eval(sys.argv[1])
    net_dims.append(1)
    digit_range = [1, 7]

    train_data, train_label, data, label = \
        mnist(noTrSamples=2000, noTsSamples=1400, \
              digit_range=digit_range, \
              noTrPerClass=1000, noTsPerClass=700)


    test_data = np.concatenate((data[:, :500], data[:, 700:1200]), axis=1)
    val_data = np.concatenate((data[:, 500:700], data[:, 1200:1400]), axis=1)
    test_label = np.concatenate((label[:, :500], label[:, 700:1200]), axis=1)
    val_label = np.concatenate((label[:, 500:700], label[:, 1200:1400]), axis=1)
    val_label[val_label == digit_range[0]] = 0
    val_label[val_label == digit_range[1]] = 1
    train_label[train_label == digit_range[0]] = 0
    train_label[train_label == digit_range[1]] = 1
    test_label[test_label == digit_range[0]] = 0
    test_label[test_label == digit_range[1]] = 1
    n_in, m = train_data.shape
    n_fin = 1
    n_h = 500

    net_dims = [n_in, n_h, n_fin]
    # initialize learning rate and num_iterations
    learning_rate = 0.1
    num_iterations = 1000

    costs, parameters,costs1 = two_layer_network(train_data, train_label,val_data, val_label, net_dims, \
                                          num_iterations=num_iterations, learning_rate=learning_rate)


    # compute the accuracy for training set and testing set
    train_Pred = classify(train_data, parameters)
    test_Pred = classify(test_data, parameters)
    val_Pred=classify(val_data,parameters)
    m1 = train_data.shape[1]
    m2 = test_data.shape[1]
    m3 = val_data.shape[1]
    trAcc = 100 - (np.sum(np.absolute(train_label - train_Pred)) / m1) * 100
    teAcc = 100 - (np.sum(np.absolute(test_label - test_Pred)) / m2) * 100
    valAcc = 100 - (np.sum(np.absolute(val_label - val_Pred)) / m3) * 100
    print("Accuracy for training set is {0:0.3f} %".format(trAcc))
    print("Accuracy for testing set is {0:0.3f} %".format(teAcc))
    print("Accuracy for validation set is {0:0.3f} %".format(valAcc))

    points = np.arange(0, 100)
    plt.plot(points, costs, label='train')
    plt.plot(points, costs1, label='validation')
    plt.xlabel('iterations')
    plt.ylabel('cost')

    plt.title("Error vs iterations")

    plt.legend()

    plt.show()
    plt.savefig("Error vs iterations")



if __name__ == "__main__":
    main()



