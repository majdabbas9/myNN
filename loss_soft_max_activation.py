from backend import np
from loss_function import LossCCE
from softmax import ActivationSoftmax
class ActivationSoftmaxLossCategoricalCrossentropy():
    def __init__(self):
        self.activation = ActivationSoftmax()
        self.loss = LossCCE()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output 
        return self.loss.calculate(self.output,y_true)
        
    def backward(self, dvalues, y_true):
        batch_size = len(dvalues)
        if len(y_true.shape) ==2:
            y_true = np.argmax(y_true,axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(batch_size), y_true] -= 1
        self.dinputs = self.dinputs / batch_size
