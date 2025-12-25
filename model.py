from nn_modeles.backend import np, to_numpy, to_tensor
from nn_modeles.nn_Layer import LayerDense
from nn_modeles.accuracy import calc_accuracy
from nn_modeles.activation_fucntions import ActivationReLU
from nn_modeles.loss_soft_max_activation import ActivationSoftmaxLossCategoricalCrossentropy
import matplotlib.pyplot as plt

class Model():
    def __init__(self, layers, optimizer):
        self.nn_layers = []
        self.activation = []
        # Create layers
        for i in range(len(layers)-1):
            self.nn_layers.append(LayerDense(layers[i], layers[i+1]))
            # Add activation for all but the last layer
            if i < len(layers) - 2:
                self.activation.append(ActivationReLU())
        
        self.optimizer = optimizer
        self.soft_LCC_loss = ActivationSoftmaxLossCategoricalCrossentropy()
        self.curr_loss = 0
        self.curr_y_pred = []
        self.number_of_layers = len(layers) - 1

    def forward(self, inputs, y_true):
        curr_input = inputs
        # Forward pass through hidden layers
        for i in range(self.number_of_layers-1):
            self.nn_layers[i].forward(curr_input)
            self.activation[i].forward(self.nn_layers[i].output)
            curr_input = self.activation[i].output
            
        # Forward pass through output layer
        self.nn_layers[-1].forward(curr_input)
        
        # Loss and prediction
        self.curr_loss = self.soft_LCC_loss.forward(self.nn_layers[-1].output, y_true)
        self.curr_y_pred = self.soft_LCC_loss.output

    def backward(self, y_true):
        # Backward pass through loss
        self.soft_LCC_loss.backward(self.soft_LCC_loss.output, y_true)
        
        # Backward pass through output layer 
        self.nn_layers[-1].backward(self.soft_LCC_loss.dinputs)
        curr_dinputs = self.nn_layers[-1].dinputs
        
        # Backward pass through hidden layers
        for i in range(self.number_of_layers-2, -1, -1):
            self.activation[i].backward(curr_dinputs)
            self.nn_layers[i].backward(self.activation[i].dinputs)
            curr_dinputs = self.nn_layers[i].dinputs

        # Update parameters
        self.optimizer.pre_update_params()
        for layer in self.nn_layers:
            self.optimizer.update_params(layer)
        self.optimizer.post_update_params()
    
    def train(self,inputs,y_true,epochs,batch_size):
        import numpy as original_numpy
        steps = len(inputs) // batch_size
        if steps * batch_size < len(inputs):
            steps += 1
        loss_stats = []
        acc_stats = []
        print(f"Training on {len(inputs)} samples, {batch_size} batch size, {epochs} epochs.")
        for epoch in range(epochs):    
            keys = np.array(range(len(inputs)))
            np.random.shuffle(keys)
            keys_cpu = original_numpy.arange(len(inputs))
            original_numpy.random.shuffle(keys_cpu)
            X_shuffled = inputs[keys_cpu]
            y_shuffled = y_true[keys_cpu]
            epoch_loss = 0
            epoch_acc = 0
            for step in range(steps):
                start = step * batch_size
                end = (step + 1) * batch_size
                # Get batch on CPU first
                batch_X_cpu = X_shuffled[start:end]
                batch_y_cpu = y_shuffled[start:end]
                batch_X = to_tensor(batch_X_cpu)
                batch_y = to_tensor(batch_y_cpu)
                self.curr_loss = 0 
                self.forward(batch_X,batch_y)
                epoch_loss += self.curr_loss
                epoch_acc +=  calc_accuracy(self.curr_y_pred,batch_y)
                self.backward(batch_y)
                # Move to Device (GPU or CPU depending on backend)
            epoch_loss /= steps
            epoch_acc /= steps
            loss_stats.append(epoch_loss)
            acc_stats.append(epoch_acc)
            if epoch % 10 == 0:
                print(f'epoch: {epoch}, acc: {epoch_acc:.3f}, loss: {epoch_loss:.3f}')
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
    
    def eval(self,inputs,y_true):
        self.curr_loss = 0
        self.curr_y_pred = []
        self.forward(inputs,y_true)
        loss = self.curr_loss
        curr_y_pred = self.curr_y_pred
        accuracy = calc_accuracy(curr_y_pred,y_true)
        print(f'Test acc: {to_numpy(accuracy):.3f}, Test loss: {to_numpy(loss):.3f}')


            
