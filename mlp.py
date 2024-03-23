import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds

class MultiLayerPerceptron():
    '''
    This class defines the multi-layer perceptron we will be using
    as the attack target.
    '''
    def __init__(self):
        self.eps = 0.1
    
    def load_params(self, params):
        '''
        This method loads the weights and biases of a trained model.
        '''
        self.W1 = params["fc1.weight"]
        self.b1 = params["fc1.bias"]
        self.W2 = params["fc2.weight"]
        self.b2 = params["fc2.bias"]
        self.W3 = params["fc3.weight"]
        self.b3 = params["fc3.bias"]
        self.W4 = params["fc4.weight"]
        self.b4 = params["fc4.bias"]
        
    def set_attack_budget(self, eps):
        '''
        This method sets the maximum L_infty norm of the adversarial
        perturbation.
        '''
        self.eps = eps
        
    def forward(self, x):
        '''
        This method finds the predicted probability vector of an input
        image x.
        
        Input
            x: a single image vector in ndarray format
        Ouput
            a vector in ndarray format representing the predicted class
            probability of x.
        '''
        W1, W2, W3, W4 = self.W1, self.W2, self.W3, self.W4
        b1, b2, b3, b4 = self.b1, self.b2, self.b3, self.b4
        
        self.z1 = np.matmul(x,W1)+b1
        self.h1 = relu(self.z1)
        self.z2 = np.matmul(self.h1,W2)+b2
        self.h2 = relu(self.z2)
        self.z3 = np.matmul(self.h2,W3)+b3
        self.h3 = relu(self.z3)
        self.z4 = np.matmul(self.h3,W4)+b4
        self.p = softmax(self.z4)
        
        return self.p
        
    def predict(self, x):
        '''
        This method takes a single image vector x and returns the 
        predicted class label of it.
        '''
        res = self.forward(x)
        return np.argmax(res)
    
    def gradient(self,x,y):
        ''' 
        This method finds the gradient of the cross-entropy loss
        of an image-label pair (x,y) w.r.t. to the image x.
        
        Input
            x: the input image vector in ndarray format
            y: the true label of x
            
        Output
            a vector in ndarray format representing
            the gradient of the cross-entropy loss of (x,y)
            w.r.t. the image x.
        '''
        p = self.forward(x)
        dLdz4 = p - y
        dz4dh3 = self.W4.T

        dh3dz3_tmp = np.zeros((self.h3.size))
        dh3dz3_tmp[self.h3 > 0] = 1
        dh3dz3 = np.diag(dh3dz3_tmp)
        dz3dh2 = self.W3.T

        dh2dz2_tmp = np.zeros((self.h2.size))
        dh2dz2_tmp[self.h2 > 0] = 1
        dh2dz2 = np.diag(dh2dz2_tmp)
        dz2dh1 = self.W2.T

        dh1dz1_tmp = np.zeros((self.h1.size))
        dh1dz1_tmp[self.h1 > 0] = 1
        dh1dz1 = np.diag(dh1dz1_tmp)
        dz1dh0 = self.W1.T # h0 is x

        return dLdz4 @ dz4dh3 @ dh3dz3 @ dz3dh2 @ dh2dz2 @ dz2dh1 @ dh1dz1 @ dz1dh0
    
    def attack(self,x,y,num_steps=1):
        '''
        This method generates the adversarial example of an
        image-label pair (x,y).
        
        Input
            x: an image vector in ndarray format, representing
               the image to be corrupted.
            y: the true label of the image x.
            
        Output
            a vector in ndarray format, representing
            the adversarial example created from image x.
        '''
        
        # Fast gradient sign method
        x_til = np.copy(x)
        for _ in range(num_steps):
            x_til += self.eps * np.sign(self.gradient(x_til,y))
        
        return np.clip(x_til, 0, 1)

    
    @staticmethod
    def perturbationLoss(r,x,y,W1, W2, W3, W4,b1, b2, b3, b4):
        '''
        This method defines the loss function to be optimized to find the optimal
        perturbation for input images
        Input: First argument must be the variable of optimization i.e. r (perturbation)
            r: perturbation array. size = (784,)
            x: input image. size = (784,)
            y: true class label. size = (1,)
        '''
        return -1 * BCE(y, fwd_attack(x+r,W1, W2, W3, W4,b1, b2, b3, b4))

    def myAttack(self,x,y):
        # cons = (
                # {'type': 'ineq', 'fun': lambda r: self.eps - np.linalg.norm(r,np.inf)}, # inf norm < eps
                # {'type': 'ineq', 'fun': lambda r: np.min(x+r)}, # lb constraint: x+r > 0
                # {'type': 'ineq', 'fun': lambda r: np.min(1-x-r)}, # ub constraint: x+r<1 i.e. 1-x-r > 0
                # )
        y = oneHotEncode(y)
        r = minimize(self.perturbationLoss, x0=np.random.normal(size=(784,)),
                     args=(x,y,self.W1, self.W2, self.W3, self.W4,self.b1, self.b2, self.b3, self.b4),
                     method='SLSQP', jac='2-point', options={'disp': False}
                    #    , constraints=cons
                       )
        # project the resulting perturbation r onto the constraint region;
        # epsilon inf norm constraint and [0,1] pixel value constraint
        pert = r.x
        pert[pert > self.eps] = self.eps
        pert = np.where(pert+x >= 1, 1,pert)
        pert = np.where(pert+x <= 0, 0,pert)
        
        return x + pert