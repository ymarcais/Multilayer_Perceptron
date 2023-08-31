# Multilayer_Perceptron
  Machine Learning Neural Network
    Multi hidden layers
    Back Propagation Neural Network
  
  - data_processing package
    - replace NaN by median
    - delete column if too much data missing
 
#FORWARD PROPAGATION

    # First layer
    Z[1] = W[1] x X + B[1]
    A[1] = 1 / (1 + exp(-Z[1]))

    # Second layer
    Z[2] = W[2] x X + B[2]
    A[2] = 1 / (1 + exp(-Z[2]))

    # binary cross-entropy error functionLog loss
    L = (-1 / N) x sum( Ylog(A[2]) + (1 - Y)log(1 - A[2]))
    with Y(n) from n = 1 to n = N

#BACK PROPAGATION
  # Calculation partiale derivative derivative with respect to W[2] && derivative with respect to B[2] 
    ∂Z[2] = (∂L / ∂A[2]) x (∂A[2] / ∂Z[2])
        (∂L / ∂A[2]) = (1 / m) x sum(((-Y / A[2]) + ((1 - y) / (1 - A[2]))))
        (∂L / ∂Z[2]) = A[2] x (1 - A[2])
    
    ∂Z[2] = (1 / m) x sum(A[2] - Y)
    ∂L / ∂W[2] = (1 / m) ∂Z[2] x A[1].T     # with 'T' for Transpose
    ∂L / ∂B[2] = (1 / m) x sum(∂Z[2])
    
    ∂Z[1] = W[2].T x ∂Z[2] x (A[1] x (1 - A[1])) # multiplication terme a terme C = A * B qui n est pas une multiplication matricielle
    ∂L / ∂W[1] = (1 / m) x ∂Z[1] x X.T
    ∂L / ∂B[1] = (1 / m) x sum(∂Z[1])

  
    
