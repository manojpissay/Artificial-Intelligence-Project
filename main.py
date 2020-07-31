import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.datasets import make_classification
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score
import math
from datetime import datetime
import warnings;

warnings.simplefilter('ignore')


class NN(object):
    def __init__(self, epsilon_init=0.12, hidden_layer_size=25, opti_method='TNC'):

        self.epsilon_init = epsilon_init
        self.hidden_layer_size = hidden_layer_size
        self.activation_func = self.sigmoid
        self.activation_func_prime = self.sigmoid_prime
        self.method = opti_method

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        sig = self.sigmoid(z)
        return sig * (1 - sig)

    def sumsqr(self, a):
        return np.sum(a ** 2)

    def rand_init(self, l_in, l_out):
        self.epsilon_init = (math.sqrt(6)) / (math.sqrt(l_in + l_out))
        return np.random.rand(l_out, l_in + 1) * 2 * self.epsilon_init - self.epsilon_init

    def pack_thetas(self, t1, t2):
        return np.concatenate((t1.reshape(-1), t2.reshape(-1)))

    def unpack_thetas(self, thetas, input_layer_size, hidden_layer_size, num_labels):
        t1_start = 0
        t1_end = hidden_layer_size * (input_layer_size + 1)
        t1 = thetas[t1_start:t1_end].reshape((hidden_layer_size, input_layer_size + 1))
        t2 = thetas[t1_end:].reshape((num_labels, hidden_layer_size + 1))
        return t1, t2

    def _forward(self, X, t1, t2):
        m = X.shape[0]
        ones = None
        if len(X.shape) == 1:
            ones = np.array(1).reshape(1, )
        else:
            ones = np.ones(m).reshape(m, 1)

        # Input layer
        a1 = np.hstack((ones, X))

        # Hidden Layer
        z2 = np.dot(t1, a1.T)
        a2 = self.activation_func(z2)
        a2 = np.hstack((ones, a2.T))

        # Output layer
        z3 = np.dot(t2, a2.T)
        a3 = self.activation_func(z3)
        return a1, z2, a2, z3, a3

    def function(self, thetas, input_layer_size, hidden_layer_size, num_labels, X, y):
        t1, t2 = self.unpack_thetas(thetas, input_layer_size, hidden_layer_size, num_labels)

        m = X.shape[0]
        Y = np.eye(num_labels)[y]

        _, _, _, _, h = self._forward(X, t1, t2)
        costPositive = -Y * np.log(h).T
        costNegative = (1 - Y) * np.log(1 - h).T
        cost = costPositive - costNegative
        J = np.sum(cost) / m
        return J

    def fit_random_hill1(self, X, y, initialw, status=False, maxiter=10000):

        num_features = X.shape[0]
        input_layer_size = X.shape[1]
        num_labels = len(set(y))

        theta1_0 = self.rand_init(input_layer_size, self.hidden_layer_size)
        theta2_0 = self.rand_init(self.hidden_layer_size, num_labels)
        thetas0 = self.pack_thetas(theta1_0, theta2_0)
        # print(len(thetas0))
        options = {'maxiter': maxiter}
        _res = optimize.fmin(self.function, initialw, maxiter=maxiter, disp=status,
                             args=(input_layer_size, self.hidden_layer_size, num_labels, X, y))
        self.t1, self.t2 = self.unpack_thetas(_res, input_layer_size, self.hidden_layer_size, num_labels)
        # returns the intermediate weights to pass to annealing
        return (_res)

    def fit_random_hill2(self, X, y, status=False, maxiter=10000):

        num_features = X.shape[0]
        input_layer_size = X.shape[1]
        num_labels = len(set(y))

        theta1_0 = self.rand_init(input_layer_size, self.hidden_layer_size)
        theta2_0 = self.rand_init(self.hidden_layer_size, num_labels)
        thetas0 = self.pack_thetas(theta1_0, theta2_0)
        # print(len(thetas0))
        options = {'maxiter': maxiter}
        _res = optimize.fmin(self.function, thetas0, maxiter=maxiter, disp=status,
                             args=(input_layer_size, self.hidden_layer_size, num_labels, X, y))
        self.t1, self.t2 = self.unpack_thetas(_res, input_layer_size, self.hidden_layer_size, num_labels)
        # returns the intermediate weights to pass to annealing
        return (_res)

    def fit_anneal1(self, X, y, status=False, maxiter=50, T0=None, bound=10,
                   learn_rate=0.1, schedule='fast', quench=0.1, ):
        """
        Fits a simulated annealing algorithm from Scipy.optimize module
        """
        num_features = X.shape[0]
        input_layer_size = X.shape[1]
        num_labels = len(set(y))

        theta1_0 = self.rand_init(input_layer_size, self.hidden_layer_size)
        theta2_0 = self.rand_init(self.hidden_layer_size, num_labels)
        thetas0 = self.pack_thetas(theta1_0, theta2_0)
        lw = [-bound] * 56
        up = [bound] * 56
        options = {'maxiter': maxiter}

        _res = optimize.dual_annealing(self.function, bounds=list(zip(lw, up)), x0=thetas0, maxiter=maxiter,
                                       initial_temp=T0, no_local_search=True,
                                       args=(input_layer_size, self.hidden_layer_size, num_labels, X, y))

        # if status:
        #        print(_res)

        self.t1, self.t2 = self.unpack_thetas(_res.x, input_layer_size, self.hidden_layer_size, num_labels)

        # np.savetxt("weights_t1.txt", self.t1, newline="\n")
        # np.savetxt("weights_t2.txt", self.t2, newline="\n")

        return (_res.x)

    def fit_anneal2(self, X, y, x0, status=False, maxiter=50, T0=None, bound=10,
                   learn_rate=0.1, schedule='fast', quench=0.1, ):
        """
        Fits a simulated annealing algorithm from Scipy.optimize module
        """
        num_features = X.shape[0]
        input_layer_size = X.shape[1]
        num_labels = len(set(y))

        theta1_0 = self.rand_init(input_layer_size, self.hidden_layer_size)
        theta2_0 = self.rand_init(self.hidden_layer_size, num_labels)
        thetas0 = self.pack_thetas(theta1_0, theta2_0)
        lw = [-bound] * 56
        up = [bound] * 56
        options = {'maxiter': maxiter}

        _res = optimize.dual_annealing(self.function, bounds=list(zip(lw, up)), x0=x0, maxiter=maxiter, initial_temp=T0,
                                       no_local_search=True,
                                       args=(input_layer_size, self.hidden_layer_size, num_labels, X, y))

        if status:
            print(_res)

        self.t1, self.t2 = self.unpack_thetas(_res.x, input_layer_size, self.hidden_layer_size, num_labels)

        # np.savetxt("weights_t1.txt", self.t1, newline="\n")
        # np.savetxt("weights_t2.txt", self.t2, newline="\n")

        return (_res.x)

    def predict(self, X):
        return self.predict_proba(X).argmax(0)

    def predict_proba(self, X):
        _, _, _, _, h = self._forward(X, self.t1, self.t2)
        return h


l = []


def generateColumns(start, end):
    for i in range(start, end + 1):
        l.extend([str(i) + 'X', str(i) + 'Y'])
    return l


eyes = generateColumns(1, 12)

# reading in the csv as a dataframe
import pandas as pd

df = pd.read_csv('EYES.csv')

# selecting the features and target
X = df[eyes]
y = df['truth_value']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Data Normalization
from sklearn.preprocessing import StandardScaler as SC

sc = SC()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# import numpy as np
X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


#Test1

print("Model-1")
print("The order of algorithms used is:")
print("1. Simulated Annealing")
print("2. Random Hill Climbing")

startTime1 = datetime.now()
#print("\nWeights that will be passed as initial guess to random_hill_climb:",res1)
nn1 = NN(hidden_layer_size=2)
res1=nn1.fit_anneal1(X_train, y_train,status=False,maxiter=10,T0=1,learn_rate=0.2,schedule='fast')
# print("Accuracy of anneal classification: ",accuracy_score(y_test, nn1.predict(X_test)))
#print("F1 score of classification: ",f1_score(y_test, nn1.predict(X_test)))
# print("Weights that will be passed as initial guess to random_hill_climb:",res1)
nn1 = NN(hidden_layer_size=2)
resSR = nn1.fit_random_hill1(X_train, y_train,res1,maxiter=40)

# print("Random hill climbing done \n")

model1=nn1
# using the learned weights to predict the target
y_pred1 = model1.predict(X_test)

# setting a confidence threshold of 0.9
y_pred_labels1 = list(y_pred1 > 0.9)

for i in range(len(y_pred_labels1)):
    if int(y_pred_labels1[i]) == 1 : y_pred_labels1[i] = 1
    else : y_pred_labels1[i] = 0

# plotting a confusion matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred_labels1)
print("Confusion Matrix:\n",cm1)
print("Accuracy:",accuracy_score(y_test, nn1.predict(X_test)))
# print("F1 score of classification: ",f1_score(y_test, nn1.predict(X_test)))
# print("\nFinal weights: ",resSR)

print("Execution time in seconds:",datetime.now() - startTime1)

print()

#Test2

print("Model-2")
print("The order of algorithms used is:")
print("1. Random Hill Climbing")
print("2. Simulated Annealing")

startTime2 = datetime.now()
nn2 = NN(hidden_layer_size=2)
res2 = nn2.fit_random_hill2(X_train, y_train, maxiter=10)

# print("Random hill climbing done. \n")
# print("Accuracy of random hill climbing classification: ",accuracy_score(y_test, nn1.predict(X_test)))
# print("F1 score of classification: ",f1_score(y_test, nn1.predict(X_test)))
# print("\nWeights that will be passed as initial guess to simulated annealing:\n ",res1)
nn2 = NN(hidden_layer_size=2)
resRA = nn2.fit_anneal2(X_train, y_train, res2, status=False, maxiter=40, T0=1, learn_rate=0.2, schedule='fast')
# using the learned weights to predict the target

model2=nn2
y_pred2 = model2.predict(X_test)

# setting a confidence threshold of 0.9
y_pred_labels2 = list(y_pred2 > 0.9)

for i in range(len(y_pred_labels2)):
    if int(y_pred_labels2[i]) == 1 : y_pred_labels2[i] = 1
    else : y_pred_labels2[i] = 0
cm2 = confusion_matrix(y_test, y_pred_labels2)
print("Confusion Matrix:\n",cm2)
print("Accuracy:", accuracy_score(y_test, nn2.predict(X_test)))
# print("F1 score of classification: ",f1_score(y_test, nn1.predict(X_test)))
# print("Final weights :",resRA)
print("Execution time in seconds:",datetime.now() - startTime2)


df_results1 = pd.DataFrame()
df_results1['Actual label'] = y_test
df_results1['Predicted value'] = nn1.predict(X_test)
df_results1['Predicted label'] = y_pred_labels1
df_results1.to_csv(r'Results1.csv')

df_results2 = pd.DataFrame()
df_results2['Actual label'] = y_test
df_results2['Predicted value'] = nn2.predict(X_test)
df_results2['Predicted label'] = y_pred_labels2
df_results2.to_csv(r'Results2.csv')
