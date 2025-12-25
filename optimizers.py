from backend import np
class OptimizerSGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

class OptimizerSGDWithMomentum:
    def __init__(self, learning_rate=1.0,alpha = 0.1):
        self.learning_rate = learning_rate
        self.alpha = alpha
    def update_params(self, layer):
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases) 

        layer.weight_momentums = self.alpha * layer.weight_momentums - self.learning_rate * layer.dweights
        layer.bias_momentums = self.alpha * layer.bias_momentums - self.learning_rate * layer.dbiases

        layer.weights += layer.weight_momentums
        layer.biases += layer.bias_momentums

class OptimizerAdaGrad:
    def __init__(self, learning_rate=1.0,smothing = 1e-7):
        self.learning_rate = learning_rate
        self.smothing = smothing
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases) 
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2
        layer.weights -= self.learning_rate * (layer.dweights / (np.sqrt(layer.weight_cache) + self.smothing))
        layer.biases -=  self.learning_rate * (layer.dbiases / (np.sqrt(layer.bias_cache) + self.smothing))
class OptimizerAdam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.smothing = smothing
        self.iter = 0
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_moving_avarge1 = np.zeros_like(layer.weights)
            layer.bias_moving_avarge1 = np.zeros_like(layer.biases)
            layer.weight_moving_avarge2 = np.zeros_like(layer.weights)
            layer.bias_moving_avarge2 = np.zeros_like(layer.biases)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases) 
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        beta1_exp_iter = self.beta1 ** (self.iter + 1)
        beta2_exp_iter = self.beta2 ** (self.iter + 1)

        layer.weight_moving_avarge1 = (beta1_exp_iter * layer.weight_moving_avarge1 + (1 - beta1_exp_iter) * layer.dweights) / (1 - beta1_exp_iter)
        layer.bias_moving_avarge1 = (beta1_exp_iter * layer.bias_moving_avarge1 + (1 - beta1_exp_iter) * layer.dbiases) / (1 - beta1_exp_iter)

        layer.weight_moving_avarge2 = (beta2_exp_iter * layer.weight_moving_avarge2 + (1 - beta2_exp_iter) * layer.weight_cache) / (1 - beta2_exp_iter)
        layer.bias_moving_avarge2 = (beta2_exp_iter * layer.bias_moving_avarge2 + (1 - beta2_exp_iter) * layer.bias_cache) / (1 - beta2_exp_iter)

        layer.weights -= self.learning_rate * (layer.weight_moving_avarge1 / (np.sqrt(layer.weight_moving_avarge2) + self.smothing))
        layer.biases -=  self.learning_rate * (layer.bias_moving_avarge1 / (np.sqrt(layer.bias_moving_avarge2) + self.smothing))
        self.iter += 1


