from backend import np, to_numpy, to_tensor
import sys
import nnfs
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from nn_Layer import LayerDense
from activation_fucntions import ActivationReLU
from accuracy import calc_accuracy
from loss_soft_max_activation import ActivationSoftmaxLossCategoricalCrossentropy
from optimizers import OptimizerSGDWithMomentum
import matplotlib.pyplot as plt

nnfs.init()

# Load MNIST data
print("Loading MNIST data... (this might take a while)")
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
X = mnist.data.astype('float32')
y = mnist.target.astype('int')

# Scale data to range [0, 1] 
X /= 255.0

print(f"Data shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
dense1 = LayerDense(784, 128)
activation1 = ActivationReLU()
dense2 = LayerDense(128, 64)
activation2 = ActivationReLU()
dense3 = LayerDense(64, 10) 
loss_activation = ActivationSoftmaxLossCategoricalCrossentropy()
optimizer = OptimizerSGDWithMomentum(learning_rate=0.1, alpha=0.9)

# Lists for tracking metrics
loss_stats = []
acc_stats = []

# Training Loop
EPOCHS = 20 
BATCH_SIZE = 128
steps = len(X_train) // BATCH_SIZE

if steps * BATCH_SIZE < len(X_train):
    steps += 1

print(f"Training on {len(X_train)} samples, {BATCH_SIZE} batch size, {EPOCHS} epochs.")

for epoch in range(EPOCHS):
    epoch_loss = 0
    epoch_acc = 0
    
    # Shuffle training data
    # We do shuffling in CPU (numpy) before converting batches to GPU
    keys = np.array(range(len(X_train)))
    np.random.shuffle(keys)
    
    # If keys is on GPU (cupy), we need to ensure X_train/y_train are accessible.
    # X_train/y_train are numpy arrays (CPU)
    # So we should shuffle in CPU numpy.
    
    # But wait, 'np' is imported from backend.
    # If backend is cupy, np.array creates a cupy array.
    # X_train is sklearn numpy array. 
    # X_train[keys] where keys is cupy array might fail or be slow.
    
    # Let's use pure numpy for shuffling indices if X_train is numpy
    import numpy as original_numpy
    keys_cpu = original_numpy.arange(len(X_train))
    original_numpy.random.shuffle(keys_cpu)
    
    X_train_shuffled = X_train[keys_cpu]
    y_train_shuffled = y_train[keys_cpu]

    for step in range(steps):
        start = step * BATCH_SIZE
        end = (step + 1) * BATCH_SIZE
        
        # Get batch on CPU first
        batch_X_cpu = X_train_shuffled[start:end]
        batch_y_cpu = y_train_shuffled[start:end]
        
        # Move to Device (GPU or CPU depending on backend)
        batch_X = to_tensor(batch_X_cpu)
        batch_y = to_tensor(batch_y_cpu)

        # Forward pass
        dense1.forward(batch_X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        dense3.forward(activation2.output)
        
        loss = loss_activation.forward(dense3.output, batch_y)
        accuracy = calc_accuracy(loss_activation.output, batch_y)

        # Convert simple metrics to CPU for accumulation (avoid GPU sync overhead/errors)
        epoch_loss += to_numpy(loss)
        epoch_acc += to_numpy(accuracy)

        # Backward pass
        loss_activation.backward(loss_activation.output, batch_y)
        dense3.backward(loss_activation.dinputs)
        activation2.backward(dense3.dinputs)
        dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update params
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.update_params(dense3)
    
    # Average metrics for the epoch
    epoch_loss /= steps
    epoch_acc /= steps
    
    loss_stats.append(epoch_loss)
    acc_stats.append(epoch_acc)

    print(f'epoch: {epoch}, acc: {epoch_acc:.3f}, loss: {epoch_loss:.3f}')

# Validation on Test Set
print('Validation')

# Move test data to device
X_test_device = to_tensor(X_test)
y_test_device = to_tensor(y_test)

dense1.forward(X_test_device)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)

loss = loss_activation.forward(dense3.output, y_test_device)
accuracy = calc_accuracy(loss_activation.output, y_test_device)
print(f'Test acc: {to_numpy(accuracy):.3f}, Test loss: {to_numpy(loss):.3f}')

# Plotting
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_stats, label='Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(acc_stats, label='Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.show()
