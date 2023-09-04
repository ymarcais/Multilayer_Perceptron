import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import sys
import os
lib_py_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib_py'))
sys.path.append(lib_py_path)
from bar_processing import bar_processing
from gradient_descent import GradientDescent
from dataclasses import dataclass

@dataclass
class Multilayer:
    gd: GradientDescent
    
    # Initialisation 
    def initialisation(self, dimensions):
        parametres = {}
        C = len(dimensions)

        for c in range(1,  C):
            parametres['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1])
            parametres['b' + str(c)] = np.random.randn(dimensions[c], 1)
        return parametres
    
    # Softmax function activation with (X - np.max(X) for better stability
    def my_softmax(self, X):
        exp_x = np.exp(X - np.max(X, axis=1, keepdims=True)) # keepdims -> broadcasting
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    # C = numero de la couche finale
    def forward_propagation(self, X, parametres):

        activations = {'A0' : X}

        C = len(parametres) // 2                
        for c in range(1, C + 1):
            Z = parametres['W' + str(c)].dot(activations['A' + str(c - 1)]) + parametres['b' + str(c)]
            activations['A' + str(c)] = self.my_softmax(Z)
      
        print(activations)
        
        return activations
    
    # Backpropagation
    def back_propagation(self, y, activations, parametres):
        m = y.shape[1]
        C = len(parametres) // 2

        dZ = activations['A' + str(C)] - y
        gradients = {}

        for c in reversed(range(1, C+1)):
            gradients['dW' + str(c)] = 1 / m * np.dot(dZ, activations['A' + str(c - 1)].T).reshape(1, -1)
            gradients['db' + str(c)] = 1 / m  * np.sum(dZ, axis = 1, keepdims=True)
            if c > 1:
                dZ = np.dot(parametres['W' + str(c)].T, dZ) * activations['A' + str(c - 1)] * (1 - activations['A' + str(c - 1)]) #multiplication matricielle entre dW et dZ et multiplications de terme a terme avec A et (1 - A)
        return gradients
    
    #Parameters update
    def update(self, gradients, parametres, learning_rate):
        C = len(parametres) // 2

        for c in range(1, C+1):
            parametres['W' + str(c)] += - learning_rate *gradients['dW' + str(c)]
            parametres['b' + str(c)] += - learning_rate *gradients['db' + str(c)]
        return parametres
    
    # binary cross-entropy error function
    def log_loss(y, A):
        return 1 / len(y) * sum(-y * np.log(A) + (1 - y) * np.log(1 - A))
    
    def predict(self, X, parametres):
        activations = self.forward_propagation(X, parametres)
        C = len(parametres) // 2
        Af = activations['A' + str(C)]
        return Af == np.max(Af, axe=0)
    
    @bar_processing
    def neural_network(self, X, y, dimensions, learning_rate, n_iter):
        train_loss = []
        train_acc = []
        
        dimensions.insert(0, X.shape[0])
        dimensions.append(y.shape[0])
        parametres = self.initialisation(dimensions)
        for i in range(n_iter):
            activations = self.forward_propagation(X, parametres)
            gradients = self.back_propagation(y, activations, parametres)
            parametres = self.update(gradients, parametres, learning_rate)

            if i% 10 == 0:
                C = len(parametres)
                train_loss.appen(self.log_loss(y, activations['A' + str(C)]))
                y_pred = self.predict(X, parametres)
                current_accuracy = gd.my_accuracy(y.flatten(), y_pred.flatten)
                train_acc.append(current_accuracy)

        self.plot_presentation(train_loss, train_acc)
        return parametres

    def plot_presentation(self, train_loss, train_acc):
        ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))
        ax[0].plot(train_loss, label='Train Loss')
        ax[0].legend()
        plt.show(train_loss)

        ax[1].plot(train_acc, label='Train Accuracy')
        ax[1].legend()
        plt.show(train_loss)

    
def main():
    gd =GradientDescent()
    ml = Multilayer(gd)
    learning_rate = 0.1
    n_iter = 1000
    dimensions = [32, 32, 32]
    dimensions.insert(0, X.shape[0])
    dimensions.append(y.shape[0])
    parametres = ml.initialisation(dimensions)
    ml.neural_network(X, y, dimensions, learning_rate, n_iter)


    
    
    
    parametres = ml.initialisation([2, 2, 2])
    
    X = np.array([0.4, 0.3])
    X = X / np.sum(X)
    y = np.array([[0.2, 0.4]])
    y = y.T
    activations = ml.forward_propagation(X, parametres )
    grad = ml.back_propagation(y, activations, parametres)
    
    for key, val in grad.items():
        print(key, val.shape)

if __name__ == "__main__":
    main()