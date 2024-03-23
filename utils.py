import numpy as numpy

def softmax(x):
    res = np.exp(x-np.max(x))
    return res/np.sum(res)

def BCE(y_true, y_pred):
     y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
     return - (y_true @ np.log(y_pred) + (1 - y_true) @ np.log(1 - y_pred))
    
def oneHotEncode(y):
     tmp = np.zeros((10,))
     tmp[y] = 1
     return tmp

def relu(x):
    return np.maximum(x, 0)

def fwd_attack(x,W1, W2, W3, W4,b1, b2, b3, b4):
        
        z1 = np.matmul(x,W1)+b1
        h1 = relu(z1)
        z2 = np.matmul(h1,W2)+b2
        h2 = relu(z2)
        z3 = np.matmul(h2,W3)+b3
        h3 = relu(z3)
        z4 = np.matmul(h3,W4)+b4
        p = softmax(z4)
        return p

