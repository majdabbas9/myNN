from backend import np, to_numpy, to_tensor
from nnfs.datasets import spiral_data
import nnfs
from nn_Layer import LayerDense
from activation_fucntions import ActivationReLU
from accuracy import calc_accuracy
from loss_soft_max_activation import ActivationSoftmaxLossCategoricalCrossentropy
from optimizers import OptimizerSGD,OptimizerSGDWithMomentum,OptimizerAdaGrad,OptimizerAdam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt    
nnfs.init()
X, y = spiral_data(samples=1000, classes=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert to backend tensor (GPU or CPU)
X_train = to_tensor(X_train)
y_train = to_tensor(y_train)
X_test = to_tensor(X_test)
y_test = to_tensor(y_test)
dense1 = LayerDense(2, 64)
activation1 = ActivationReLU()
dense2 = LayerDense(64, 3)
loss_activation = ActivationSoftmaxLossCategoricalCrossentropy()
optimizer = OptimizerAdam(learning_rate=0.005, decay=1e-3)

loss_stats = []
acc_stats = []

for epoch in range(10001):
    dense1.forward(X_train)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output,y_train)
    accuracy = calc_accuracy(loss_activation.output,y_train)

    if not epoch % 100:
        loss_stats.append(to_numpy(loss))
        acc_stats.append(to_numpy(accuracy))
        print(f'epoch: {epoch}, ' +f'acc: {accuracy:.3f}, ' +f'loss: {loss:.3f}, ' + f'lr: {optimizer.learning_rate}')

    loss_activation.backward(loss_activation.output,y_train)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

# Validate the model
print('Validation')
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y_test)
accuracy = calc_accuracy(loss_activation.output, y_test)
print(f'validation, acc: {to_numpy(accuracy):.3f}, loss: {to_numpy(loss):.3f}')

# Plotting
plt.figure()
plt.plot(loss_stats, label='Loss')
plt.plot(acc_stats, label='Accuracy')
plt.legend()
plt.show()
