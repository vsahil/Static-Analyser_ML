import numpy as np

import h5py, time, sys
sys.path.append("../")
import scipy
from scipy import ndimage

start_time = time.time()
train_set_x = np.array([
    [1,1,1,1],[0,1,1,1],[0,0,1,1]
])

train_set_y = np.array([
    [1,1,1],[1,1,0],[1,1,1]
])

numberOfFeatures = 4
numberOfTrainingExamples = 3

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

w = np.zeros((numberOfTrainingExamples , 1))
b = 0
A = sigmoid(np.dot(w.T , train_set_x))
print(A)

import check
np.multiply(train_set_y,A)

def initialize_with_zeros(numberOfTrainingExamples):
    w = np.zeros((numberOfTrainingExamples , 1))
    b = 0
    return w, b

def propagate(w, b, X, Y):

    m = X.shape[1]

    A = sigmoid(np.dot(w.T , X) + b)

    cost = -(1/m)*np.sum(np.multiply(Y,np.log(A)) + np.multiply((1-Y),np.log(1-A)), axis=1)

    dw =  ( 1 / m ) *   np.dot( X, ( A - Y ).T )    # consumes ( A - Y )
    db =  ( 1 / m ) *   np.sum( A - Y )    # consumes ( A - Y ) again

#     cost = np.squeeze(cost)

    grads = {"dw": dw,
             "db": db}

    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = True):

    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - (learning_rate * dw)
        b = b - (learning_rate * db)

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 10000 == 0:
            print(cost)

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs

def model(X_train, Y_train, num_iterations, learning_rate = 0.5, print_cost = False):

    w, b = initialize_with_zeros(numberOfTrainingExamples)

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost = True)
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_train = sigmoid(np.dot(w.T , X_train) + b)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))

model(train_set_x, train_set_y, num_iterations = 1, learning_rate = 0.0001, print_cost = True)
