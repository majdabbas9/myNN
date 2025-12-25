from .backend import np
def calc_accuracy(y_pred,y_true):
    predections = np.argmax(y_pred,axis=1)
    if len(y_true.shape) == 2:
        y_true = np.argmax(y_true,axis=1)
    return np.mean(predections == y_true)
