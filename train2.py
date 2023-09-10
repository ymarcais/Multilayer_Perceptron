import numpy as np

# Define the architecture
input_size = 3  # Change this to match your input size
hidden_layer1_size = 50
hidden_layer2_size = 50
output_size = 2

# Initialize weights and biases
def initialize_parameters():
    np.random.seed(0)
    W1 = np.random.randn(hidden_layer1_size, input_size) * 0.01
    b1 = np.zeros((hidden_layer1_size, 1))
    W2 = np.random.randn(hidden_layer2_size, hidden_layer1_size) * 0.01
    b2 = np.zeros((hidden_layer2_size, 1))
    W3 = np.random.randn(output_size, hidden_layer2_size) * 0.01
    b3 = np.zeros((output_size, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

# Define the activation function (e.g., sigmoid)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Implement forward propagation
def forward_propagation(X, parameters):
    W1, b1, W2, b2, W3, b3 = parameters
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}

# Implement the cost function (e.g., mean squared error)
def compute_cost(A3, Y):
    m = Y.shape[1]
    cost = -np.sum(Y * np.log(A3) + (1 - Y) * np.log(1 - A3)) / m
    return cost

# Implement backward propagation
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1, W2, W3 = parameters["W1"], parameters["W2"], parameters["W3"]
    A1, A2, A3 = cache["A1"], cache["A2"], cache["A3"]
    
    dZ3 = A3 - Y
    dW3 = np.dot(dZ3, A2.T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = dA2 * A2 * (1 - A2)
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * A1 * (1 - A1)
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    
    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}
    return gradients

# Update parameters using gradient descent
def update_parameters(parameters, gradients, learning_rate=0.01):
    W1, b1, W2, b2, W3, b3 = parameters
    dW1, db1, dW2, db2, dW3, db3 = gradients
    
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    
    updated_parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    return updated_parameters

# Define the training function
def train(X, Y, num_epochs=1000, learning_rate=0.01):
    parameters = initialize_parameters()
    costs = []
    
    for epoch in range(num_epochs):
        cache = forward_propagation(X, parameters)
        cost = compute_cost(cache["A3"], Y)
        gradients = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, gradients, learning_rate)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Cost {cost}")
            costs.append(cost)
    
    return parameters, costs

# Example data
X = np.random.randn(input_size, 1000)
Y = np.random.randint(0, 2, (output_size, 1000))

# Train the model
trained_parameters, cost_history = train(X, Y)

# You can now use the trained_parameters for prediction.
