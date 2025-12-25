from model import Model
from nn_modeles.optimizers import OptimizerAdam
from nn_modeles.backend import np, to_numpy, to_tensor
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import nnfs

nnfs.init()

# 1. Load Data
print("Loading MNIST data...")
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
X = mnist.data.astype('float32')
y = mnist.target.astype('int')
X /= 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors (if using backend with tensors)
X_train = to_tensor(X_train)
y_train = to_tensor(y_train)
X_test = to_tensor(X_test)
y_test = to_tensor(y_test)

# 2. Define Model
print("Defining model...")
model = Model([784,128,64,10],OptimizerAdam(learning_rate=0.005, decay=1e-3))
model.train(X_train,y_train,20,128)
model.eval(X_test,y_test)

