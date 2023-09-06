import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
import sys
import os
lib_py_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib_py'))
sys.path.append(lib_py_path)
from bar_processing import bar_processing
from gradient_descent import GradientDescent
from tqdm import tqdm
import time

# Bar processing decorator
def local_bar_processing(func):
    def wrapper(*args, **kwargs):
        progress_bar = tqdm(total=100, dynamic_ncols=True)
        result = func(*args, **kwargs)
        progress_bar.close()
        return result
    return wrapper

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
    
    #get y and X from train.csv
    def get_y_X(self, train_data_path):
        dataset = pd.read_csv(train_data_path, header=None)
        y = pd.DataFrame(dataset.iloc[:, 0]).values
       # print("y shape:",y.shape)
        X = pd.DataFrame(dataset.iloc[:, 1:]).values
        #print("X shape:",X.shape)
        return y, X
    
    # Softmax function activation with (X - np.max(X) for better stability
    def my_softmax(self, X):
        exp_x = np.exp(X - np.max(X, axis=1, keepdims=True)) # keepdims -> broadcasting
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    # C = numero de la couche finale
    def forward_propagation(self, X, parametres):

        activations = {'A0' : X}

        C = len(parametres) // 2                
        for c in range(1, C + 1):
            #print("parametres['W' + str(c)]:", parametres['W' + str(c)].shape)
            #print("activations['A' + str(c - 1)", activations['A' + str(c - 1)].shape)
            #print("parametres['W' + str(c)]:", parametres['b' + str(c)].shape)
            Z = parametres['W' + str(c)].dot(activations['A' + str(c - 1)]) + parametres['b' + str(c)]
            print("Z:", Z)
            activations['A' + str(c)] = self.my_softmax(Z)
      
        '''for key, value in activations.items():
            print(f'{key}: {value.shape}')'''
        
        return activations
    
    # Backpropagation
    def back_propagation(self, y, activations, parametres):
        m = y.shape[1]
        #print("y shape:", y.shape)
        C = len(parametres) // 2
        dZ = activations['A' + str(C)] - y
        gradients = {}

        for c in reversed(range(1, C+1)):
            #gradients['dW' + str(c)] = 1 / m * np.dot(dZ, activations['A' + str(c - 1)].T).reshape(1, -1)
            gradients['dW' + str(c)] = 1 / m * np.dot(dZ, activations['A' + str(c - 1)].T)
            gradients['db' + str(c)] = 1 / m  * np.sum(dZ, axis = 1, keepdims=True)
            if c > 1:
                dZ = np.dot(parametres['W' + str(c)].T, dZ) * activations['A' + str(c - 1)] * (1 - activations['A' + str(c - 1)]) #multiplication matricielle entre dW et dZ et multiplications de terme a terme avec A et (1 - A)
        return gradients
    
    #Parameters update
    def update(self, gradients, parametres, learning_rate):
        C = len(parametres) // 2

        for c in range(1, C+1):
            #gradient_reshaped = np.array(gradients['dW' + str(c)].reshape(461, 461))
            #print(f"shape W{C}:", parametres['W' + str(c)].shape)
            #print(f"shape dW{C}:", gradients['dW' + str(c)].shape)
            #print(f"shape db{C}:", gradients['db' + str(c)].shape)
            parametres['W' + str(c)] += - learning_rate *gradients['dW' + str(c)]
            parametres['b' + str(c)] += - learning_rate *gradients['db' + str(c)]
        return parametres
    
    # binary cross-entropy error function
    def log_loss(self, y, A):
        log_loss_ = 1 / len(y) * sum(-y * np.log(A) + (1 - y) * np.log(1 - A))
        #print("A:", A)
        #print("sum:", sum(-y * np.log(A) + (1 - y) * np.log(1 - A)))
        #print("len(y):", len(y))
        #print("log_loss:", log_loss_)
        return log_loss_
    
    def predict(self, X, parametres):
        activations = self.forward_propagation(X, parametres)
        C = len(parametres) // 2
        Af = activations['A' + str(C)]
        Af = np.max(Af, axis=0)
        Af = np.array([Af])
        print("Af:", Af)
        print("Af:", Af.shape)
        return Af
    
    #New way to calculate accuracy
    def my_accuracy(self,y ,y_predict):
        y = np.array([y])
        y_predict = np.array([y_predict])
        #print("y:", y.shape)

        correct_matches = np.sum(y - y_predict)
        total_sample = y.shape[0]
        my_accuracy = 100 * correct_matches / total_sample 
        print("my_accuracy:", my_accuracy)
        return my_accuracy
    
    @local_bar_processing
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
                C = len(parametres) // 2
                #print("activations:", activations)
                train_loss.append(self.log_loss(y, activations['A' + str(C)]))
                y_pred = self.predict(X, parametres)
                #print("y_predict:", y_pred.shape)
                current_accuracy = self.my_accuracy(y, y_pred)
                #current_accuracy = self.my_accuracy(y.flatten(), y_pred.flatten())
                train_acc.append(current_accuracy)

        self.plot_presentation(train_loss, train_acc)
        return parametres

    def plot_presentation(self, train_loss, train_acc):
        iterations = list(range(0, len(train_loss) * 10, 10))
        #train_loss =np.array([train_loss])
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 4))
        #ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))
        #print("Train Loss:", train_loss)
        #print("Train Loss:", train_loss.shape)
        ax[0].plot(train_loss, label='Train Loss')
        ax[0].set_title("Train Loss")
        #ax[0].legend()

        ax[1].plot(train_acc, label='Train Accuracy')
        ax[1].legend()
        plt.show()
    
def main():
    thetas = []
    alpha = []
    max_iter = 0
    gd =GradientDescent(thetas, alpha, max_iter)
    ml = Multilayer(gd)
    learning_rate = 0.1
    n_iter = 1000
    dimensions = [461, 461]
    train_data_path = "train_data.csv"
    y, X = ml.get_y_X(train_data_path)
    dimensions.insert(0, X.shape[0])
    dimensions.append(y.shape[0])


    parametres = ml.initialisation(dimensions)
    ml.neural_network(X, y, dimensions, learning_rate, n_iter)
  
    X = X / np.sum(X)
    #y = y.T
    activations = ml.forward_propagation(X, parametres )
    grad = ml.back_propagation(y, activations, parametres)
    
    for key, val in grad.items():
        print(key, val.shape)

if __name__ == "__main__":
    main()