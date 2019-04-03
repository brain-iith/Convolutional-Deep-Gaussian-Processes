import numpy as np
import tensorflow as tf
import gpflow
import pickle

from gpflow.likelihoods import MultiClass
from gpflow.kernels import RBF, White
from gpflow.models.svgp import SVGP
from gpflow.training import AdamOptimizer

from scipy.stats import mode
from scipy.cluster.vq import kmeans2

from dgp import DGP
import convkernels as ckern

import time

import sys

old_stdout = sys.stdout

log_file = open("logs/4layered.log","w")

sys.stdout = log_file

num_classes = 1
def get_data():
    d = np.load('data/convex.npz')
    X, Y, Xtest, Ytest = d['X'], d['Y'], d['Xtest'], d['Ytest']
    
    return X.astype(float), Y.astype(float), Xtest.astype(float), Ytest.astype(float)

X, Y, Xs, Ys = get_data()
X = X.astype(np.float32,copy=False)
Xs = Xs.astype(np.float32,copy=False)
Y = Y.astype(np.float32,copy=False)
Ys = Ys.astype(np.float32,copy=False)

X_val = X
Y_val = Y


# In[ ]:

print("Intializing inducing variables")
M = 100
Z = kmeans2(X, M, minit='points')[0]


# In[ ]:


minibatch_size = 1000
epochs = 200
iter_epochs = int(X.shape[0] / minibatch_size)
iterations = int(epochs*iter_epochs)


# In[ ]:

def make_dgp(L):

    # kernels = [ckern.WeightedColourPatchConv(RBF(25*1, lengthscales=10., variance=10.), [28, 28], [5, 5], colour_channels=1)]
    kernels = [RBF(784, lengthscales=10., variance=1.)]
    for l in range(L-1):
        kernels.append(RBF(50, lengthscales=10.))
    model = DGP(X, Y, Z, kernels, gpflow.likelihoods.Bernoulli(), 
                minibatch_size=minibatch_size,
                num_outputs=num_classes)
    
    # start things deterministic 
    for layer in model.layers[:-1]:
        layer.q_sqrt = layer.q_sqrt.value * 1e-5 
    
    return model

# print("building 2 layered model", flush=True)
# m_dgp2 = make_dgp(2)

# print("building 3 layered model")
# m_dgp3 = make_dgp(3)

print("building 4 layered model")
m_dgp4 = make_dgp(4)


S = 100
def assess_model_dgp(model, X_batch, Y_batch):
    l, m, v = model.predict_density_nd_y(X_batch, Y_batch, S)
    a = (mode(np.where(m>0.5,1,0), 0)[0].reshape(Y_batch.shape).astype(int)==Y_batch.astype(int))
    return l, a


# In[5]:


def batch_assess(model, assess_model, X, Y):
    #here 100 is batch size
    n_batches = max(int(len(X)/100), 1)
    lik, acc = [], []
    for X_batch, Y_batch in zip(np.split(X, n_batches), np.split(Y, n_batches)):
        l, a = assess_model(model, X_batch, Y_batch)
        lik.append(l)
        acc.append(a)
    lik = np.concatenate(lik, 0)
    acc = np.array(np.concatenate(acc, 0), dtype=float)
    return np.average(lik), np.average(acc)



def minimize(model, lr, iterations, var_list=None, callback=None):

    session = model.enquire_session()
    
    times = []
    nlpp = []
    accuracy = []
    adam = AdamOptimizer(lr).make_optimize_action(model)
    with session.as_default():
        for _i in range(iterations):
            if(_i%iter_epochs==0):
                print("epoch---" + str( int(_i/iter_epochs)) + "done." , flush=True)
                if not(_i/iter_epochs==0):
                    print ("epoch time === ",time.time() - start, flush=True)
                    times.append(time.time() - start)
                start = time.time()
                model.anchor(session)
                param_dict = model.read_trainables()
                f = open("models/wconv" + str(len(model.layers)) + ".pkl","wb")
                pickle.dump(param_dict,f)
                f.close()
                if (int(_i/iter_epochs)%5==0):
                    l, a = batch_assess(model, assess_model_dgp, X_val, Y_val)
                    print('training lik: {:.4f}, training acc {:.4f}'.format(l, a), flush=True)
                    nlpp.append(l)
                    accuracy.append(a)
            adam()
            model.anchor(session)

    l, a = batch_assess(model, assess_model_dgp, Xs, Ys)
    print('training lik: {:.4f}, training acc {:.4f}'.format(l, a), flush=True)
    nlpp.append(l)
    accuracy.append(a)

    model.anchor(session)
            
    return np.asarray(nlpp), np.asarray(accuracy), np.asarray(times)



print("training 4 layered model")
start_gb = time.time()
nlpp, accuracy, times = minimize(m_dgp4, 0.01, iterations, callback=None)
print ("training time === ",time.time() - start_gb)
np.save("nlpp/wconv4",nlpp)
np.save("acc/wconv4",accuracy)
np.save("time/wconv4",times)
param_dict = m_dgp4.read_trainables()
f = open("models/wconv" + str(len(m_dgp4.layers)) + ".pkl","wb")
pickle.dump(param_dict,f)
f.close()


sys.stdout = old_stdout

log_file.close()