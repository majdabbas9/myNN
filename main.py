from nn_modeles.backend import np, to_numpy, to_tensor
from nnfs.datasets import spiral_data
import nnfs
from nn_modeles.optimizers import OptimizerSGD,OptimizerSGDWithMomentum,OptimizerAdaGrad,OptimizerAdam
from sklearn.model_selection import train_test_split 
from model import Model 
nnfs.init()
X, y = spiral_data(samples=1000, classes=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert to backend tensor (GPU or CPU)
X_train = to_tensor(X_train)
y_train = to_tensor(y_train)
X_test = to_tensor(X_test)
y_test = to_tensor(y_test)
optimizer = OptimizerAdam(learning_rate=0.005, decay=1e-3)

model = Model([2,64,3],optimizer)
model.train(X_train,y_train,10001,len(X_train))
model.eval(X_test,y_test)