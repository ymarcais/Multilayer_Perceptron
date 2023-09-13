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
        #print("dimensions:", dimensions)

        np.random.seed(1)

        for k in range(1, C):
            if k == 1:
                parametres['W' + str(k)] = np.random.randn(dimensions[k], X.shape[0])
                parametres['b' + str(k)] = np.random.randn(dimensions[k], 1)
                #print("X.shape[1]:", X.shape[1])
                #print("k:",k)
                #print(f"W{k}:", parametres['W' + str(k)].shape)
                #print(f"b{k}:", parametres['b' + str(k)].shape)
            elif k < C - 1:
                parametres['W' + str(k)] = np.random.randn(dimensions[k], dimensions[k-1])
                parametres['b' + str(k)] = np.random.randn(dimensions[k], 1)
                #print("k:",k)
                #print("dimension[1]:", dimensions[1])
                #print(f"b{k}",parametres['b' + str(k)].shape)
            else:
                parametres['W' + str(C-1)] = np.random.randn(dimensions[C-1], dimensions[k-1])
                parametres['b' + str(C-1)] = np.random.randn(dimensions[C-1], 1)
                #print("dimension[1]:", dimensions[1])
                #print(f"W{k}:", parametres['W' + str(k)].shape)
                #print(f"b{k}:", parametres['b' + str(k)].shape)
            
            #print("parametres:", parametres)
        return parametres
    
    #get y and X from train.csv
    def get_y_X(self, train_data_path):
        dataset = pd.read_csv(train_data_path, header=None)
        y = pd.DataFrame(dataset.iloc[2:, 0]).values
        y = y.reshape(1, -1)
        X = pd.DataFrame(dataset.iloc[2:, 1:]).values
        #print("y shape begins:", y.shape)
        return y, X.T
    
    # Softmax function activation with (X - np.max(X) for better stability
    def my_softmax(self, X):
        exp_x = np.exp(X - np.max(X, axis=0, keepdims=True)) # keepdims -> broadcasting
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def relu(self, X):
        return np.max(X, 0)
  
    # C = numero de la couche finale
    def forward_propagation(self, X, parametres):
        #print("X shape", X.shape)
        activations = {'A0' : X}

        #print("X shape:", activations['A0'].shape)

        C = len(parametres) // 2
        #print("len parametres:", len(parametres))            
        for c in range(1, C + 1):
            #print(f"W{c}:", parametres['W' + str(c)].shape)
            #print(f"X.T shape{c}", X.shape)
            #print(f"A{c-1}:", activations['A' + str(c - 1)].shape)
            #print(f"b{c}:", parametres['b' + str(c)].shape)
            #Z = (parametres['W' + str(c)].dot(activations['A' + str(c - 1)])) + parametres['b' + str(c)]
            #Z = (activations['A' + str(c - 1)]).dot(parametres['W' + str(c)].T) + parametres['b' + str(c)].T
            Z = parametres['W' + str(c)].dot(activations['A' + str(c - 1)]) + parametres['b' + str(c)]
            #print(f"Z{c}:", Z.shape, f"W{c}", parametres['W' + str(c)].shape, f"[A{c - 1}", activations['A' + str(c - 1)].shape)
            #activations['A' + str(c)] = 1 / (1 + np.exp(-Z))
            #activations['A' + str(c)] = self.my_softmax(Z)
            if c == C - 1:
                #print("C:", C)
                activations['A' + str(c)] = self.my_softmax(Z)
                #activations['A' + str(c)] = 1 / (1 + np.exp(-Z))
                #activations['A' + str(c)] = (activations['A' + str(c)] > 0.5).astype(int)
                #print(activations['A' + str(c)])
                #activations['A' + str(c)] = np.sum(activations['A' + str(c)])
            else:
                activations['A' + str(c)] = 1 / (1 + np.exp(-Z))
                #print(f"W{c}:", parametres['W' + str(c)].shape)
                #activations['A' + str(c)] = self.relu(Z)
                #activations['A' + str(c)] = self.my_softmax(Z)
                #print("activation with softmax:", activations['A' + str(c)].shape)
                #activations['A' + str(c)] = (activations['A' + str(c)] > 0.5).astype(int)
                #print(activations['A' + str(c)])
                #print(f"Z{c}:", Z.shape)
                
        #print("AC shape:",activations['A' + str(c)].shape)
      
        '''for key, value in activations.items():
            print(f'{key}: {value.shape}')'''

        return activations
        
    # Backpropagation
    def back_propagation(self, y, activations, parametres):
        m = y.shape[1]
        #print("m:", m)
        C = len(parametres) // 2
        #print(f"A{C} shape:", activations['A' + str(C)].shape, " // y shape", y.shape)
        dZ = activations['A' + str(C)] - y
        #print(f"A{C}:",activations['A' + str(C)].shape)
        #print(y.shape)
        #print("dZ:", dZ.shape)
        
        gradients = {}

        for c in reversed(range(1, C+1)):
            #print("c:",c)
            #print("// dZ.T:", dZ.T.shape, "// A C - 1:", activations['A' + str(c - 1)].shape)
            if c == C:
                #print("C:", C)
                #print("c:", c)
                gradients['dW' + str(c)] = 1 / m * np.dot( dZ, activations['A' + str(c - 1)].T)
                #print(f"dW{c}:", gradients['dW' + str(c)].shape, "// dZ.T:", dZ.T.shape, "// A C - 1:", activations['A' + str(c - 1)].shape)
                gradients['db' + str(c)] = 1 / m  * np.sum(dZ, axis=1, keepdims=True)
            elif c < C and c > 1:
                #print("C:", C)
                #print("c:", c)
                gradients['dW' + str(c)] = 1 / m * np.dot(dZ, activations['A' + str(c - 1)].T)
                #print(f"dW{c}:", gradients['dW' + str(c)].shape, "// dZ.T:", dZ.T.shape, "// A C - 1:", activations['A' + str(c - 1)].shape)
                gradients['db' + str(c)] = 1 / m  * np.sum(dZ, axis=1, keepdims=True)
            else:
                #print("C:", C)
                #print("c:", c)
                gradients['dW' + str(c)] = 1 / m * np.dot(dZ, activations['A' + str(c - 1)].T)
                #print(f"dW{c}:", gradients['dW' + str(c)].shape, "// dZ.T:", dZ.T.shape, "// A C - 1:", activations['A' + str(c - 1)].shape)
                gradients['db' + str(c)] = 1 / m  * np.sum(dZ, axis=1, keepdims=True)                

            if c > 1:
                #print(f"Wc:", (parametres['W' + str(c)].T).shape, "// dZ:", dZ.shape)
                #print("A c - 1:", activations['A' + str(c - 1)].shape)
                #a_terme = activations['A' + str(c - 1)].T * (1 - activations['A' + str(c - 1)].T)
                #print("A terme:", a_terme.shape)
                #toto = np.dot(parametres['W' + str(c)].T, dZ.T)
                #print("toto:", toto.shape)
                #multiplication matricielle entre dW et dZ et multiplications de terme a terme avec A et (1 - A)
                dZ = np.dot(parametres['W' + str(c)].T, dZ) * activations['A' + str(c - 1)] * (1 - activations['A' + str(c - 1)])
                #print("dZ toto:", dZ.shape)
        return gradients
    
    #Parameters update
    def update(self, gradients, parametres, learning_rate):
        C = len(parametres) // 2

        for c in range(1, C+1):
            if c == 1:
                #print("C update:", C)
                #print("c update:", c)
                #print(f"W{c}:", parametres['W' + str(c)].shape, f"// dW{c}", gradients['dW' + str(c)].shape)
                parametres['W' + str(c)] += - learning_rate *gradients['dW' + str(c)]
                #print(f"b{c}:", parametres['b' + str(c)].shape, "// db{c}", gradients['db' + str(c)].shape)
                parametres['b' + str(c)] += - learning_rate *gradients['db' + str(c)]
            elif c > 1:
                #print("C update:", C)
                #print("c update:", c)
                #print("learning_rate:", learning_rate)
                #print(f"W{c}:", parametres['W' + str(c)].shape, f"// dW{c}", gradients['dW' + str(c)].shape)
                parametres['W' + str(c)] += - learning_rate * gradients['dW' + str(c)]
                #print(f"b{c}:", parametres['b' + str(c)].shape, "// db{c}", gradients['db' + str(c)].shape)
                parametres['b' + str(c)] += - learning_rate *gradients['db' + str(c)]

        return parametres
    
    # binary cross-entropy error function
    def log_loss(self, y, A):
        A = A[0]
        #print("y shape", y.shape)
        leny = y.shape[1]
        #print("A shape", A.shape)
        #print("leny", leny)
        epsilon = 1e-15
        log_loss_ = 1 / leny * np.sum(-y * np.log(np.maximum(A, epsilon)) + (1 - y) * np.log(np.maximum(1 - A, epsilon)))

        #print("log_loss:", log_loss_.shape)
        return log_loss_
    
    def predict(self, X, parametres):
        activations = self.forward_propagation(X, parametres)
        C = len(parametres) // 2
        #print("AActivation:", activations['A' + str(C)])
        Af = activations['A' + str(C)]
        Af = (Af > 0.5).astype(int)
        #print("AActivation:", activations['A' + str(C)])
        #print("Af:", Af)
        #print("Af:", Af)
        return Af
    
    #New way to calculate accuracy
    def my_accuracy(self, y, y_predict):
        #print("y_predic", y_predict.shape)
        correct_matches = 0
        total_samples = 0
        y_predict_first_column = y_predict[0, :]
        correct_matches = np.sum(y == y_predict_first_column)
        #print("y shape", y.shape)
        #print("y_predict shape", y_predict.shape)
        #print("correct_matches:", correct_matches)
        #print("y_predict", y_predict)
        total_samples = y.shape[1]
        #print("total sample:", total_samples)
        #print("y :", y.shape)
        #print("len y:", len(y))
        #print(y)
        #print("total matches:", correct_matches)
        my_accuracy = 100 * correct_matches / total_samples
        return my_accuracy
    
     #calculate accuracy in multilabel classification
    def f1_score(self, y, y_pred):
        # Get the unique class labels
        y_prediction = np.array([y_pred[0,:]])
        print("y:", y.shape)
        print("y_predict:", y_prediction.shape)
        unique_classes = np.unique(np.concatenate((y, y_pred)))
        print("unique_classes", unique_classes)

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

   
    #@local_bar_processing
    def neural_network(self, X, y, parametres, learning_rate, n_iter):

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
            if j%100 == 0:
                print("j:",j)
            j += 1
            activations = self.forward_propagation(X, parametres)
            gradients = self.back_propagation(y, activations, parametres)
            parametres = self.update(gradients, parametres, learning_rate)
            #Af = activations['A' + str(C)]
            #print("Af", Af)
            y_pred = self.predict(X, parametres)
            training_history[i, 0] = (self.log_loss(y, y_pred))
            
            #print("y_pred:", y_pred.shape)
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
        #legend_label = f"{len_dimensions} layers \n"
        legend = plt.legend(label_lines,loc='center left', bbox_to_anchor=(1.0, 0.5), borderpad=1)
        legend.get_frame().set_linewidth(0)  # Adjust the legend box border width
        #legend.get_frame().set_edgecolor('black')  # Set the legend box border color
        legend.get_frame().set_facecolor('lightblue')  # Set the legend box background color
        legend.get_frame().set_alpha(0.3)
        for text in legend.get_texts():
            text.set_fontsize(10)  # Adjust the legend font size
            text.set_color('blue')

    
    #plot double charts
    def plot_presentation(self, training_history, f1_accuracy, len_dimensions):
        final_row = training_history[-1]
        final_value = final_row[-1]
        title_lines =   [
		f"Training set accuracy: {final_value:.2f}%"
        #f"Test set accuracy : {f1_accuracy['test_accuracy']:.2f}%"
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
    print("test_accuracy:", test_accuracy)
    ml.plot_presentation(training_history, f1_accuracy, len_dimensions)

  
    '''X = X / np.sum(X)
    activations = ml.forward_propagation(X, parametres )
    grad = ml.back_propagation(y, activations, parametres)'''
    
    '''for key, val in grad.items():
        print(key, val.shape)'''
if __name__ == "__main__":
    main()