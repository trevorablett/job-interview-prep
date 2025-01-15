import numpy as np
import matplotlib.pyplot as plt

# Define the network structure
input_size = 2
hidden_size = 2
output_size = 1

# Initialize weights and biases
np.random.seed(42)
W1 = np.random.rand(input_size, hidden_size)  # Weights for input to hidden
b1 = np.random.rand(hidden_size)             # Biases for hidden layer
W2 = np.random.rand(hidden_size, output_size)  # Weights for hidden to output
b2 = np.random.rand(output_size)             # Bias for output layer

# Define the activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Training data (X: inputs, y: true outputs)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Learning rate
lr = 0.1

# Training loop
for epoch in range(20000):
    # Forward pass
    z1 = np.dot(X, W1) + b1  # Input to hidden layer
    a1 = sigmoid(z1)         # Activation at hidden layer
    z2 = np.dot(a1, W2) + b2 # Input to output layer
    a2 = sigmoid(z2)         # Activation at output layer (final prediction)

    # Compute loss (Mean Squared Error)
    loss = np.mean((y - a2) ** 2)

    # Backpropagation
    d_loss_a2 = 2 * (a2 - y) / y.shape[0]          # Derivative of loss w.r.t a2
    d_a2_z2 = sigmoid_derivative(z2)               # Derivative of activation function
    d_z2_a1 = W2                                   # Derivative of z2 w.r.t a1
    d_a1_z1 = sigmoid_derivative(z1)               # Derivative of activation function

    # Gradients for weights and biases
    d_z2_W2 = a1                                   # Gradient of z2 w.r.t W2
    d_loss_W2 = np.dot(d_z2_W2.T, d_loss_a2 * d_a2_z2)
    d_loss_b2 = np.sum(d_loss_a2 * d_a2_z2, axis=0)

    d_loss_a1 = np.dot(d_loss_a2 * d_a2_z2, d_z2_a1.T)
    d_loss_W1 = np.dot(X.T, d_loss_a1 * d_a1_z1)
    d_loss_b1 = np.sum(d_loss_a1 * d_a1_z1, axis=0)

    # Update weights and biases
    W2 -= lr * d_loss_W2
    b2 -= lr * d_loss_b2
    W1 -= lr * d_loss_W1
    b1 -= lr * d_loss_b1

    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Final output after training
print("Trained outputs:")
print(a2)

# view the function outputs to see non-linear decision boundary
# take a grid of points from x, y = [-1, 2], get outputs, color those above .5 as green, below .5 as red
grid_size = 100
boundary = 0.5

x = np.linspace(-1, 2, grid_size)
# y = np.linspace(-1, 2, grid_size)
y = np.linspace(-1, 2, grid_size)
xx, yy = np.meshgrid(x, y)

# input needs to be N x 2
# grid_X = np.stack([xx.flatten(), yy.flatten()]).T
grid_X = np.stack([yy.flatten()[::-1], xx.flatten()]).T
z1 = np.dot(grid_X, W1) + b1  # Input to hidden layer
a1 = sigmoid(z1)         # Activation at hidden layer
z2 = np.dot(a1, W2) + b2 # Input to output layer
a2 = sigmoid(z2)         # Activation at output layer (final prediction)

# outs = a2.reshape(grid_size, grid_size)
outs = a2.reshape(grid_size, grid_size)

plt.imshow(outs, extent=[-1, 2, -1, 2])
plt.scatter(X[:, 0], X[:, 1])
plt.colorbar()
plt.show()