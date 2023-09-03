import numpy as np
import pandas as pd

class Multilayer:
    
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
    
def main():
    ml = Multilayer()

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