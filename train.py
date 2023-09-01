import numpy as np
import pandas as pd

class Multilayer:
    
    # Initialisation 
    def initialisation(self, dimensions):
        parametres = {}
        C = len(dimensions)

        for c in range(1, C):
            parametres['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1])
            parametres['b' + str(c)] = np.random.randn(dimensions[c], 1)
        return parametres
    
    # Softmax function activation
    def my_softmax(self, X):
        exp_x = np.exp(X - np.max(X))
        sum_exp_x = np.sum(exp_x)
        return exp_x / sum_exp_x 
    
    # C = numero de la couche finale
    def forward_propagation(self, X, parametres):

        activations = {'A0' : X}

        C = len(parametres) // 2                
                
        for c in range(1, C + 1):
            Z = parametres['W' + str(c)].dot(activations['A' + str(c - 1)]) + parametres['b' + str(c)]
            activations['A' + str(c)] = self.my_softmax(Z)
         
        print(activations)
        
        return activations
    
def main():
    ml = Multilayer()

    parametres = ml.initialisation([3, 2, 2])
    X = np.array([0.4, 0.3, 0.2])
    X = X / np.sum(X)
    activations = ml.forward_propagation(X, parametres )

if __name__ == "__main__":
    main()