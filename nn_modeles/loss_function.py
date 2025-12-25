from .backend import np
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class LossCCE(Loss):
    def forward(self,y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if (len(y_true.shape) == 1): # for lables that its the class number which means that the numbers they contain are the correct class numbers
            confidence = y_pred_clipped[range(len(y_pred)), y_true] 
        elif (len(y_true.shape) == 2): # for one-hot encoded [0,1,0] for class 2 etc
            confidence = np.sum(y_pred_clipped * y_true, axis=1)
        return -np.log(confidence)
        
    def backward(self, dvalues, y_true):
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true,axis=1)
        self.dinputs = -y_true / dvalues
        self.dinputs /= len(y_true)

