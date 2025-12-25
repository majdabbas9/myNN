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
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))
        else:
            self.current_learning_rate = self.learning_rate

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum  with current gradients
        layer.weight_momentums = self.beta1 * \
            layer.weight_momentums + (1 - self.beta1) * layer.dweights
        layer.bias_momentums = self.beta1 * \
            layer.bias_momentums + (1 - self.beta1) * layer.dbiases
        
        # Update cache with squared current gradients
        layer.weight_cache = self.beta2 * layer.weight_cache + \
            (1 - self.beta2) * layer.dweights**2
        layer.bias_cache = self.beta2 * layer.bias_cache + \
            (1 - self.beta2) * layer.dbiases**2

        # Bias correction
        iteration = self.iterations + 1
        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta1 ** iteration)
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta1 ** iteration)
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta2 ** iteration)
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta2 ** iteration)

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         weight_momentums_corrected / \
                         (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        bias_momentums_corrected / \
                        (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


