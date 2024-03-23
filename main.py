from __future__ import print_function
from mlp import *
from utils import *
import matplotlib.pyplot as plt


X_test = np.load("./data/X_test.npy")
Y_test = np.load("./data/Y_test.npy")

params = {}
param_names = ["fc1.weight", "fc1.bias",
               "fc2.weight", "fc2.bias",
               "fc3.weight", "fc3.bias",
               "fc4.weight", "fc4.bias"]

for name in param_names:
    params[name] = np.load("./data/"+name+'.npy')
    
clf = MultiLayerPerceptron()
clf.load_params(params)

##### Cross entropy maximization based generation #############

# visualize adversarial image based on my method
x,y = X_test[12], Y_test[12]
x_til = clf.myAttack(x,y)
print ("This is an image of Number", y)
pixels = x_til.reshape((28,28))
plt.imshow(pixels,cmap="gray")

# generate adversarial examples for various values of epsilon
nTest = 100
list_acc_opt = []
for eps in [0.05, 0.1,0.15,0.2]:
    print(eps)
    clf.set_attack_budget(eps)
    Y_pred_opt = np.zeros(nTest)
    for i in range(nTest):
        if i%10==0:
            print(i)
        x, y = X_test[i], Y_test[i]
        x_til = clf.myAttack(x,y)
        Y_pred_opt[i] = clf.predict(x_til)
    acc_opt = np.sum(Y_pred_opt == Y_test[:nTest])*1.0/nTest
    list_acc_opt.append(acc_opt)


# benchmark - Fast Gradient Sign Method

# Visualize FGSM perturbed image
x = clf.attack(X_test[12], Y_test[12], 3)
y = Y_test[12]
print ("This is an image of Number", y)
pixels = x.reshape((28,28))
plt.imshow(pixels,cmap="gray")

# 
list_acc_fgsm = []
for eps in [0.05,0.1,0.15,0.2]:
    print(eps)
    clf.set_attack_budget(eps)
    Y_pred_FGSM = np.zeros(nTest)
    for i in range(nTest):
        x, y = X_test[i], Y_test[i]
        x_til = clf.attack(x,y,3)
        Y_pred_FGSM[i] = clf.predict(x_til)
    acc_fgsm = np.sum(Y_pred_FGSM == Y_test[:nTest])*1.0/nTest
    list_acc_fgsm.append(acc_fgsm)


# plot results

plt.plot([0.05,0.1,0.15,0.2], list_acc_fgsm, marker='x',label="3-step FGSM")
plt.plot([0.05,0.1,0.15,0.2], list_acc_opt, marker='x',label="Cross-entropy maximization")
plt.title('Test accuracy, various methods of generating adversarial images',fontsize=8)
plt.ylabel('Test accuracy')
plt.xlabel('Epsilon')
plt.legend()
plt.show()
