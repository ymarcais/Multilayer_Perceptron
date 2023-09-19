import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
import sys
import os

class Multilayer:
    
    '''Initialisation : dictionary for W: 'Weights' and b: 'bias' '''
    def initialisation(self, X, dimensions):
        parametres = {}
        C = len(dimensions)

        np.random.seed(1)

        for k in range(1, C):
            if k == 1:
                parametres['W' + str(k)] = np.random.randn(dimensions[k], X.shape[0])
                parametres['b' + str(k)] = np.random.randn(dimensions[k], 1)
            else:
                parametres['W' + str(k)] = np.random.randn(dimensions[k], dimensions[k-1])
                parametres['b' + str(k)] = np.random.randn(dimensions[k], 1)
        return parametres
    
    #get_y_X take train csv file path and retur X.T the input dataset and y the output dataset
    def get_y_X(self, train_data_path):
        dataset = pd.read_csv(train_data_path, header=None)
        y = pd.DataFrame(dataset.iloc[2:, 0]).values
        y = y.reshape(1, -1)
        X = pd.DataFrame(dataset.iloc[2:, 1:]).values
        return y, X.T
    
    # my_softmax function activation with X as input dataset and returns output probabilities
    def my_softmax(self, X):
        exp_x = np.exp(X - np.max(X, axis=0, keepdims=True)) # keepdims -> broadcasting
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    # ReLU is an activation function if the input value is greater than zero, it passes through unchanged; otherwise, it becomes zero
    def relu(self, X):
        return np.maximum(0, X)
  
    # C = numero de la couche finale
    def forward_propagation(self, X, parametres):
        activations = {'A0' : X}
        C = len(parametres) // 2
        for c in range(1, C + 1):
            Z = parametres['W' + str(c)].dot(activations['A' + str(c - 1)]) + parametres['b' + str(c)]
            if c == C - 1:
                activations['A' + str(c)] = self.my_softmax(Z)
                #activations['A' + str(c)] = 1 / (1 + np.exp(-Z))
            else:
                activations['A' + str(c)] = 1 / (1 + np.exp(-Z))
                #activations['A' + str(c)] = self.relu(Z)
                #activations['A' + str(c)] = self.my_softmax(Z)
        return activations
        
    # Backpropagation
    def back_propagation(self, y, activations, parametres):
        m = y.shape[1]
        C = len(parametres) // 2
        dZ = activations['A' + str(C)] - y
        gradients = {}

        for c in reversed(range(1, C+1)):
            if c == C:
                gradients['dW' + str(c)] = 1 / m * np.dot( dZ, activations['A' + str(c - 1)].T)
                gradients['db' + str(c)] = 1 / m  * np.sum(dZ, axis=1, keepdims=True)
            elif c < C and c > 1:
                gradients['dW' + str(c)] = 1 / m * np.dot(dZ, activations['A' + str(c - 1)].T)
                gradients['db' + str(c)] = 1 / m  * np.sum(dZ, axis=1, keepdims=True)
            else:
                gradients['dW' + str(c)] = 1 / m * np.dot(dZ, activations['A' + str(c - 1)].T)
                gradients['db' + str(c)] = 1 / m  * np.sum(dZ, axis=1, keepdims=True)                
            if c > 1:
                dZ = np.dot(parametres['W' + str(c)].T, dZ) * activations['A' + str(c - 1)] * (1 - activations['A' + str(c - 1)])
        return gradients
    
    #Parameters update with C as number of layers
    def update(self, gradients, parametres, learning_rate):
        C = len(parametres) // 2

        for c in range(1, C+1):
            parametres['W' + str(c)] += - learning_rate * gradients['dW' + str(c)]
            parametres['b' + str(c)] += - learning_rate * gradients['db' + str(c)]
        return parametres
    
    # binary cross-entropy error function
    def log_loss(self, y, A):
        A = A[0]
        leny = y.shape[1]
        epsilon = 1e-15
        log_loss_ = 1 / leny * np.sum(-y * np.log(np.maximum(A, epsilon)) + (1 - y) * np.log(np.maximum(1 - A, epsilon)))
        return log_loss_
    
    # make the prediction from X as output layer and the parametres dictionary and return activation matrix as output
    def predict(self, X, parametres):
        activations = self.forward_propagation(X, parametres)
        C = len(parametres) // 2
        Af = activations['A' + str(C)]
        Af = (Af > 0.5).astype(int)
        return Af
    
    #New way to calculate accuracy
    def my_accuracy(self, y, y_predict):
        correct_matches = 0
        total_samples = 0
        y_predict_first_column = y_predict[0, :]
        correct_matches = np.sum(y == y_predict_first_column)
        total_samples = y.shape[1]
        my_accuracy = 100 * correct_matches / total_samples
        return my_accuracy
    
     #f1_score calculate accuracy in multilabel classification
    def f1_score(self, y, y_pred):
        y_prediction = np.array([y_pred[0,:]])
        unique_classes = np.unique(np.concatenate((y, y_pred)))

        # Initialize arrays to store precision, recall, and f1-score for each class
        precision_scores = []
        recall_scores = []
        f1_scores = []

        # Iterate over each class
        for cls in unique_classes:
            # Compute true positives, false positives, and false negatives for the current class
            true_positives = np.sum((y == cls) & (y_prediction == cls))
            false_positives = np.sum((y != cls) & (y_prediction == cls))
            false_negatives = np.sum((y == cls) & (y_prediction != cls))

            # Calculate precision and recall for the current class
            precision = true_positives / (true_positives + false_positives + 1e-10)
            recall = true_positives / (true_positives + false_negatives + 1e-10)

            # Calculate the F1-score for the current class
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

            # Append the scores to the respective lists
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        macro_precision = 100 *np.mean(precision_scores)
        macro_recall = 100 *np.mean(recall_scores)
        macro_f1_score = 100 * np.mean(f1_scores)
        return macro_f1_score, macro_recall, macro_precision
   
    # mother funciton that launch the neural network
    def neural_network(self, X, y, parametres, learning_rate, n_iter):

        np.random.seed(1)
     
        training_history = np.zeros((int(n_iter), 2))
        C = len(parametres) // 2    
        j = 0
        for i in range(n_iter):
            if j%100 == 0:
                print("j:",j)
            j += 1
            activations = self.forward_propagation(X, parametres)
            gradients = self.back_propagation(y, activations, parametres)
            parametres = self.update(gradients, parametres, learning_rate)
            y_pred = self.predict(X, parametres)
            training_history[i, 0] = (self.log_loss(y, y_pred))
            training_history[i, 1] = (self.my_accuracy(y, y_pred))
        return training_history, parametres


    #plot colors
    def plot_format(self):
        plt.gca().spines['bottom'].set_color('blue')
        plt.gca().spines['top'].set_color('blue')
        plt.gca().spines['right'].set_color('blue')
        plt.gca().spines['left'].set_color('blue')
        plt.tick_params(axis='both', colors='blue')


    # plot legend parameters
    def plot_legend(self, len_dimensions, f1_accuracy):
        label_lines =   [
            f"{len_dimensions} layers \n"
            f"F1_score: {f1_accuracy['macro_f1_score']:.2f}% \n"
            f"Recall: {f1_accuracy['macro_recall']:.2f}% \n"
            f"Precision: {f1_accuracy['macro_precision']:.2f}% \n"
            f"Test_accuracy: {f1_accuracy['test_accuracy']:.2f}% \n"
                       ]
        legend = plt.legend(label_lines,loc='center left', bbox_to_anchor=(1.0, 0.5), borderpad=1)
        legend.get_frame().set_linewidth(0)
        legend.get_frame().set_facecolor('lightblue')
        legend.get_frame().set_alpha(0.3)
        for text in legend.get_texts():
            text.set_fontsize(10)
            text.set_color('blue')

    
    #plot double charts
    def plot_presentation(self, training_history, f1_accuracy, len_dimensions):
        final_row = training_history[-1]
        final_value = final_row[-1]
        title_lines =   [
		f"Training set accuracy: {final_value:.2f}%"
                        ]
        title = '\n'.join(title_lines)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.subplots_adjust(left=0.05)
        plt.plot(training_history[:, 0], label='train loss', color='red', linewidth=3)
        plt.title(f"Training set cost loss:", color='blue')
        self.plot_format()

        plt.subplot(1, 2, 2)
        plt.subplots_adjust(right=0.80)
        plt.plot(training_history[:, 1], label='train acc', color='red', linewidth=3)
        plt.title(title, color='blue')
        self.plot_format()
        self.plot_legend(len_dimensions, f1_accuracy)        

        plt.show()

    
def main():

    ml = Multilayer()
    train_data_path = "train_data.csv"
    test_data_path = "test_data.csv"
    y_train, X_train = ml.get_y_X(train_data_path)
    y_test, X_test = ml.get_y_X(test_data_path)
    f1_accuracy = {}

    learning_rate = 0.01
    n_iter = 5000
    input_layer = [X_train.shape[0]]
    hidden_layers = [50, 50, 50, 50, 50]
    out_put_layer = [2]
    dimensions = input_layer + hidden_layers + out_put_layer
    len_dimensions = len(dimensions)
    parametres = ml.initialisation(X_train, dimensions)
    training_history, parametres = ml.neural_network(X_train, y_train, parametres, learning_rate, n_iter)
    input_layer = [X_test.shape[0]]
    dimensions = input_layer + hidden_layers + out_put_layer
    y_test_predict = ml.predict(X_test, parametres)
    test_accuracy = ml.my_accuracy(y_test, y_test_predict)
    macro_f1_score, macro_recall, macro_precision = ml.f1_score(y_test, y_test_predict)
    f1_accuracy['macro_f1_score'] = macro_f1_score
    f1_accuracy['macro_recall'] = macro_recall
    f1_accuracy['macro_precision'] = macro_precision
    f1_accuracy['test_accuracy'] = test_accuracy
    ml.plot_presentation(training_history, f1_accuracy, len_dimensions)

if __name__ == "__main__":
    main()
