import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
import sys
import os
'''lib_py_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib_py'))
sys.path.append(lib_py_path)
from bar_processing import bar_processing
from gradient_descent import GradientDescent'''
#from tqdm import tqdm
#import time

'''# Bar processing decorator
def local_bar_processing(func):
    def wrapper(*args, **kwargs):
        progress_bar = tqdm(total=100, dynamic_ncols=True)
        result = func(*args, **kwargs)
        progress_bar.close()
        return result
    return wrapper'''

@dataclass
class Multilayer:
    
    # Initialisation 
    def initialisation(self, X, dimensions):
        parametres = {}
        C = len(dimensions)
        print("dimensions:", dimensions)

        np.random.seed(1)

        for k in range(1, C):
            if k == 1:
                parametres['W' + str(k)] = np.random.randn(X.shape[1], dimensions[k])
                parametres['b' + str(k)] = np.random.randn(dimensions[k-1], 1)
                print("k:",k)
                print(f"W{k}:", parametres['W' + str(k)].shape)
                print(f"b{k}:", parametres['b' + str(k)].shape)
                print(f"b{k}",parametres['b' + str(k)].shape)
            elif k < C - 1:
                parametres['W' + str(k)] = np.random.randn(dimensions[k], dimensions[k-1])
                parametres['b' + str(k)] = np.random.randn(X.shape[0], 1)
                print(f"b{k}",parametres['b' + str(k)].shape)
            else:
                parametres['W' + str(C-1)] = np.random.randn(dimensions[1], dimensions[C-1])
                parametres['b' + str(C-1)] = np.random.randn(X.shape[0], 1)
                print(f"W{k}:", parametres['W' + str(k)].shape)
                print(f"b{k}:", parametres['b' + str(k)].shape)
                print(f"b{k}",parametres['b' + str(k)].shape)
            
            #print("parametres:", parametres)
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
        #print("X shape", X.shape)
        activations = {'A0' : X}

        print("X shape:", activations['A0'].shape)

        C = len(parametres) // 2
        #print("len parametres:", len(parametres))            
        for c in range(1, C + 1):
            #print(f"W{c}:", parametres['W' + str(c)].shape)
            #print(f"X.T shape{c}", X.shape)
            print(f"A{c-1}:", activations['A' + str(c - 1)].shape)
            #print(f"b{c}:", parametres['b' + str(c)].shape)
            #Z = (parametres['W' + str(c)].dot(activations['A' + str(c - 1)])) + parametres['b' + str(c)]
            #Z = (activations['A' + str(c - 1)]).dot(parametres['W' + str(c)].T) + parametres['b' + str(c)].T
            Z = activations['A' + str(c - 1)].dot(parametres['W' + str(c)]) + parametres['b' + str(c)]
            print(f"Z{c}:", Z.shape, f"W{c}", parametres['W' + str(c)].shape, f"[A{c - 1}", activations['A' + str(c - 1)].shape)
            #activations['A' + str(c)] = 1 / (1 + np.exp(-Z))
            #activations['A' + str(c)] = self.my_softmax(Z)
            if c == C:
                #print("C:", C)
                #activations['A' + str(c)] = self.my_softmax(Z)
                activations['A' + str(c)] = 1 / (1 + np.exp(-Z))
                #activations['A' + str(c)] = (activations['A' + str(c)] > 0.5).astype(int)
                #print(activations['A' + str(c)])
                #activations['A' + str(c)] = np.sum(activations['A' + str(c)])
            else:
                #activations['A' + str(c)] = 1 / (1 + np.exp(-Z))
                #print(f"W{c}:", parametres['W' + str(c)].shape)
                #activations['A' + str(c)] = self.relu(Z)
                activations['A' + str(c)] = self.my_softmax(Z)
                #print("activation with softmax:", activations['A' + str(c)].shape)
                #activations['A' + str(c)] = (activations['A' + str(c)] > 0.5).astype(int)
                #print(activations['A' + str(c)])
                #print("Z{c}:", Z.shape)
                
        #print("AC shape:",activations['A' + str(c)].shape)
      
        '''for key, value in activations.items():
            print(f'{key}: {value.shape}')'''

        return activations
        
    # Backpropagation
    def back_propagation(self, y, activations, parametres):
        m = y.shape[1]
        #print("m:", m)
        C = len(parametres) // 2
        print(f"A{C} shape:", activations['A' + str(C)].shape, " // y shape", y.shape)
        dZ = activations['A' + str(C)] - y
        #print("dZ:", dZ.shape)
        
        gradients = {}

        for c in reversed(range(1, C+1)):
            #print("c:",c)
            #print("// dZ.T:", dZ.T.shape, "// A C - 1:", activations['A' + str(c - 1)].shape)
            if c == C:
                #print("C:", C)
                #print("c:", c)
                gradients['dW' + str(c)] = 1 / m * np.dot(activations['A' + str(c - 1)].T, dZ)
                print(f"dW{c}:", gradients['dW' + str(c)].shape, "// dZ.T:", dZ.T.shape, "// A C - 1:", activations['A' + str(c - 1)].shape)
                gradients['db' + str(c)] = 1 / m  * np.sum(dZ, axis=1, keepdims=True)
            elif c < C and c > 1:
                #print("C:", C)
                #print("c:", c)
                gradients['dW' + str(c)] = 1 / m * np.dot(activations['A' + str(c - 1)].T, dZ)
                print(f"dW{c}:", gradients['dW' + str(c)].shape, "// dZ.T:", dZ.T.shape, "// A C - 1:", activations['A' + str(c - 1)].shape)
                gradients['db' + str(c)] = 1 / m  * np.sum(dZ, axis=1, keepdims=True)
            else:
                #print("C:", C)
                #print("c:", c)
                gradients['dW' + str(c)] = 1 / m * np.dot(activations['A' + str(c - 1)].T, dZ)
                print(f"dW{c}:", gradients['dW' + str(c)].shape, "// dZ.T:", dZ.T.shape, "// A C - 1:", activations['A' + str(c - 1)].shape)
                gradients['db' + str(c)] = 1 / m  * np.sum(dZ, axis=1, keepdims=True)                

            if c > 1:
                #print(f"Wc:", (parametres['W' + str(c)].T).shape, "// dZ:", dZ.shape)
                #print("A c - 1:", activations['A' + str(c - 1)].shape)
                #a_terme = activations['A' + str(c - 1)].T * (1 - activations['A' + str(c - 1)].T)
                #print("A terme:", a_terme.shape)
                #toto = np.dot(parametres['W' + str(c)].T, dZ.T)
                #print("toto:", toto.shape)
                #multiplication matricielle entre dW et dZ et multiplications de terme a terme avec A et (1 - A)
                dZ = np.dot(dZ,parametres['W' + str(c)].T) * activations['A' + str(c - 1)] * (1 - activations['A' + str(c - 1)])
                #print("dZ toto:", dZ.shape)
        return gradients
    
    #Parameters update
    def update(self, gradients, parametres, learning_rate):
        C = len(parametres) // 2

        for c in range(1, C+1):
            if c == 1:
                print("C update:", C)
                print("c update:", c)
                print(f"W{c}:", parametres['W' + str(c)].shape, f"// dW{c}", gradients['dW' + str(c)].shape)
                parametres['W' + str(c)] += - learning_rate *gradients['dW' + str(c)]
                print(f"b{c}:", parametres['b' + str(c)].shape, "// db{c}", gradients['db' + str(c)].shape)
                parametres['b' + str(c)] += - learning_rate *gradients['db' + str(c)]
            elif c > 1:
                print("C update:", C)
                print("c update:", c)
                print("learning_rate:", learning_rate)
                print(f"W{c}:", parametres['W' + str(c)].shape, f"// dW{c}", gradients['dW' + str(c)].shape)
                parametres['W' + str(c)] += - learning_rate * gradients['dW' + str(c)]
                print(f"b{c}:", parametres['b' + str(c)].shape, "// db{c}", gradients['db' + str(c)].shape)
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
        Af = np.mean(activations['A' + str(C)], axis=1, keepdims=True)
        Af = (Af > 0.5).astype(int)
        print("Af:", Af.shape)
        print("Af:", Af)
        return Af
    
    #New way to calculate accuracy
    def my_accuracy(self, y, y_predict):
        #print("y_predic", y_predict.shape)
        correct_matches = np.sum(y == y_predict)
        #print("y_predict", y_predict)
        total_samples = len(y)
        #print("len y:", len(y))
        my_accuracy = 100 * correct_matches / total_samples
        return my_accuracy
    
     #calculate accuracy in multilabel classification
    '''def f1_score(self, y, y_pred):
        # Get the unique class labels
        unique_classes = np.unique(np.concatenate((y, y_pred)))

        # Initialize arrays to store precision, recall, and f1-score for each class
        precision_scores = []
        recall_scores = []
        f1_scores = []

        # Iterate over each class
        for cls in unique_classes:
            # Compute true positives, false positives, and false negatives for the current class
            true_positives = np.sum((y == cls) & (y_pred == cls))
            false_positives = np.sum((y != cls) & (y_pred == cls))
            false_negatives = np.sum((y == cls) & (y_pred != cls))

            # Calculate precision and recall for the current class
            precision = true_positives / (true_positives + false_positives + 1e-10)
            recall = true_positives / (true_positives + false_negatives + 1e-10)

            # Calculate the F1-score for the current class
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

            # Append the scores to the respective lists
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        # Calculate the overall F1-score (macro-average)
        macro_f1_score = np.mean(f1_scores)

        return macro_f1_score'''

   
    #@local_bar_processing
    def neural_network(self, X, y, parametres, learning_rate, n_iter):
        train_loss = []
        train_acc = []
        
        #dimensions = list(hidden_layers)
        #dimensions.insert(0, X.shape[0])
        #print("X[1]", X.shape[0])
        #dimensions.append(y.shape[0])
        np.random.seed(1)
        #parametres = self.initialisation(dimensions)
        
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
            #print("y_pred:", y_pred.shape)
            training_history[i, 1] = (self.my_accuracy(y, y_pred))

            if i% 10 == 0:
                C = len(parametres) // 2
                train_loss.append(self.log_loss(y, activations['A' + str(C)]))
                y_pred = self.predict(X, parametres)
                #print("y_pred:", y_pred)
                current_accuracy = self.my_accuracy(y, y_pred)
                #print("accuracy:", current_accuracy)
                train_acc.append(current_accuracy)

        return training_history

    def plot_presentation(self, training_history):

        final_row = training_history[-1]
        final_value = final_row[-1]
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(training_history[:, 0], label='train loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(training_history[:, 1], label='train acc')
        plt.title(f"Train accuracy reaches: {final_value:.2f}%")

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
    #dimensions.insert(0, X.shape[0])
    #dimensions.append(y.shape[1])
    #print("dimensions shape [0]:",y.shape[0])
    #print("dimensions shape [1]:",y.shape[1])
    #print("dimensions:", dimensions)
    parametres = ml.initialisation(X, dimensions)
    training_history = ml.neural_network(X, y, parametres, learning_rate, n_iter)
    ml.plot_presentation(training_history)

  
    '''X = X / np.sum(X)
    activations = ml.forward_propagation(X, parametres )
    grad = ml.back_propagation(y, activations, parametres)'''
    
    '''for key, val in grad.items():
        print(key, val.shape)'''
if __name__ == "__main__":
    main()