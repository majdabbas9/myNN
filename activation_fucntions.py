from backend import np

class ActivationReLU :
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class ActivationSigmoid :
    def sigmoid(self,inputs):
        return 1 / (1+np.exp(-inputs))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.sigmoid(inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        tmp = self.sigmoid(self.inputs) 
        self.dinputs *= tmp * (1 - tmp) 
