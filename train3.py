import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
import sys
import os

@dataclass
class Multilayer:
    
    # Initialisation 
    def initialisation(self, dimensions):
        parametres = {}
        C = len(dimensions)
        print("dimensions:", dimensions)

        np.random.seed(1)

        for k in range(1,  C-1):
            parametres['W' + str(k)] = np.random.randn(dimensions[k], dimensions[k-1])
            parametres['b' + str(k)] = np.random.randn(dimensions[k], 1)
            print("k:",k)
            print(f"W{k}:", parametres['W' + str(k)].shape)
            print(f"b{k}:", parametres['b' + str(k)].shape)
        parametres['W' + str(C-1)] = np.random.randn(dimensions[C-1], dimensions[k])
        parametres['b' + str(C-1)] = np.random.randn(dimensions[C-1], 1)
            
        return parametres
    
    #get y and X from train.csv
    def get_y_X(self, train_data_path):
        dataset = pd.read_csv(train_data_path, header=None)
        y = pd.DataFrame(dataset.iloc[2:, 0]).values
        X = pd.DataFrame(dataset.iloc[2:, 1:]).values
        #print("y shape begins:", y.shape)
        return y, X
    
    # Softmax function activation with (X - np.max(X) for better stability
    def my_softmax(self, X):
        exp_x = np.exp(X - np.max(X, axis=1, keepdims=True)) # keepdims -> broadcasting
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def relu(self, X):
        return np.max(X, 0)
  
    # C = numero de la couche finale
    def forward_propagation(self, X, parametres):
        activations = {'A0' : X}

        C = len(parametres) // 2
        
        for c in range(1, C + 1):
            Z = parametres['W' + str(c)].dot(activations['A' + str(c - 1)]) + parametres['b' + str(c)]
            if c == C:
                activations['A' + str(c)] = 1 / (1 + np.exp(-Z))
            else:
                activations['A' + str(c)] = self.my_softmax(Z)

        return activations
        
    # Backpropagation
    def back_propagation(self, y, activations, parametres):
        m = y.shape[1]
        C = len(parametres) // 2
        print(f"A{C} shape:", activations['A' + str(C)].shape, " // y shape", y.shape)
        dZ = activations['A' + str(C)] - y
        
        gradients = {}

        for c in reversed(range(1, C+1)):
            gradients['dW' + str(c)] = 1 / m * np.dot(dZ, activations['A' + str(c - 1)].T)
            gradients['db' + str(c)] = 1 / m  * np.sum(dZ, axis=1, keepdims=True)
            if c > 1:
                dZ = np.dot(parametres['W' + str(c)].T, dZ) * activations['A' + str(c - 1)] * (1 - activations['A' + str(c - 1)])
        return gradients
    
    #Parameters update
    def update(self, gradients, parametres, learning_rate):
        C = len(parametres) // 2

        for c in range(1, C+1):
            print(f"W{c}:", parametres['W' + str(c)].shape, "// dW{c}", gradients['dW' + str(c)].shape)
            parametres['W' + str(c)] += - learning_rate *gradients['dW' + str(c)]
            parametres['b' + str(c)] += - learning_rate *gradients['db' + str(c)]
        return parametres
    
    # binary cross-entropy error function
    def log_loss(self, y, A):
        epsilon = 1e-15
        log_loss_ = 1 / len(y) * np.sum(-y * np.log(np.maximum(A, epsilon)) + (1 - y) * np.log(np.maximum(1 - A, epsilon)))

        #print("log_loss:", log_loss_.shape)
        return log_loss_
    
    def predict(self, X, parametres):
        activations = self.forward_propagation(X, parametres)
        C = len(parametres) // 2
        Af = activations['A' + str(C)]
        return Af >= 0.5
    
    #New way to calculate accuracy
    def my_accuracy(self, y, y_predict):
        y_predict = y_predict
        correct_matches = np.sum(y == y_predict)
        total_samples = len(y)
        my_accuracy = correct_matches / total_samples
        return my_accuracy
   
    #@local_bar_processing
    def neural_network(self, X, y, parametres, learning_rate, n_iter):
        train_loss = []
        train_acc = []
        np.random.seed(1)        
        training_history = np.zeros((int(n_iter), 2))
        C = len(parametres) // 2    
        j = 0
        for i in range(n_iter):
            if j%10 == 0:
                print("j:",j)
            j += 1
            activations = self.forward_propagation(X, parametres)
            gradients = self.back_propagation(y, activations, parametres)
            parametres = self.update(gradients, parametres, learning_rate)
            Af = activations['A' + str(C)]

            training_history[i, 0] = (self.log_loss(y, Af))
            y_pred = self.predict(X, parametres)
            training_history[i, 1] = (self.my_accuracy(y, y_pred))

            if i% 10 == 0:
                C = len(parametres) // 2
                train_loss.append(self.log_loss(y, activations['A' + str(C)]))
                y_pred = self.predict(X, parametres)
                current_accuracy = self.my_accuracy(y, y_pred)
                train_acc.append(current_accuracy)

        return training_history

    def plot_presentation(self, training_history):

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(training_history[:, 0], label='train loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(training_history[:, 1], label='train acc')
        plt.legend()
        plt.show()
    
def main():

    ml = Multilayer()
    learning_rate = 0.01
    n_iter = 200
    input_layer = [464]
    hidden_layers = [50, 50]
    out_put_layer = [2]
    dimensions = input_layer + hidden_layers + out_put_layer
    train_data_path = "train_data.csv"
    y, X = ml.get_y_X(train_data_path)
    parametres = ml.initialisation(dimensions)
    training_history = ml.neural_network(X, y, parametres, learning_rate, n_iter)
    ml.plot_presentation(training_history)

if __name__ == "__main__":
    main()