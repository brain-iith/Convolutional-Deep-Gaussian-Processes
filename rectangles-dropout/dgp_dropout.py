import numpy as np
import tensorflow as tf
import gpflow
import pickle
from keras.datasets import cifar10
from keras.utils import to_categorical

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

log_file = open("logs/dgp_dropout.log","w")

sys.stdout = log_file

num_classes = 2
def get_data():
    d = np.load('data/rectangles_im.npz')
    X, Y, Xtest, Ytest = d['X'], d['Y'], d['Xtest'], d['Ytest']
    
    return X.astype(float), Y.astype(float), Xtest.astype(float), Ytest.astype(float)

X, Y, Xs, Ys = get_data()

X_val = Xs[:10000]
Y_val = Ys[:10000]

Xs = Xs[10000:]
Ys = Ys[10000:]

# In[ ]:

print("Intializing inducing variables")
M = 500
Z = kmeans2(X, M, minit='points')[0]


# In[ ]:


minibatch_size = 500
epochs = 400
iter_epochs = int(X.shape[0] / minibatch_size)
iterations = int(epochs*iter_epochs)


# In[ ]:

def make_dgp(L):

    # kernels = [ckern.WeightedColourPatchConv(RBF(25*1, lengthscales=10., variance=10.), [28, 28], [5, 5], colour_channels=1)]
    kernels = [RBF(784,lengthscales=10., variance=10.)]
    for l in range(L-1):
        kernels.append(RBF(50, lengthscales=10., variance=10.))
    model = DGP(X, Y, Z, kernels, gpflow.likelihoods.MultiClass(num_classes), 
                minibatch_size=minibatch_size,
                num_outputs=num_classes, dropout = 0.5)
    
    # start things deterministic 
    for layer in model.layers[:-1]:
        layer.q_sqrt = layer.q_sqrt.value * 1e-5 
    
    return model

print("building 2 layered model", flush=True)
m_dgp2 = make_dgp(2)

print("building 3 layered model")
m_dgp3 = make_dgp(3)

print("building 4 layered model")
m_dgp4 = make_dgp(4)


# In[ ]:


S = 100
def assess_model_dgp(model, X_batch, Y_batch):
    l, m, v = model.predict_density_nd_y(X_batch, Y_batch, S)
    a = (mode(np.argmax(m, 2), 0)[0].reshape(Y_batch.shape).astype(int)==Y_batch.astype(int))
    return l, a


# In[ ]:


def batch_assess(model, assess_model, X, Y):
    #here 50 is batch size
    n_batches = max(int(len(X)/100), 1)
    lik, acc = [], []
    for X_batch, Y_batch in zip(np.split(X, n_batches), np.split(Y, n_batches)):
        l, a = assess_model(model, X_batch, Y_batch)
        lik.append(l)
        acc.append(a)
    lik = np.concatenate(lik, 0)
    acc = np.array(np.concatenate(acc, 0), dtype=float)
    return np.average(lik), np.average(acc)

# In[ ]:

def minimize(model, lr, iterations, var_list=None, callback=None):

    session = model.enquire_session()
    # loss_val = []
    times = []
    nlpp = []
    accuracy = []
    # inputs = tf.placeholder(tf.float64, shape=( X_val.shape[0], X_val.shape[1]))
    # labels = tf.placeholder(tf.float64, shape=( Y_val.shape[0], Y_val.shape[1]))
    adam = AdamOptimizer(lr).make_optimize_action(model)
    # natgrad = NatGradOptimizer(gamma).make_optimize_action(model, var_list=var_list)
    with session.as_default():
        for _i in range(iterations):
            if(_i%iter_epochs==0):
                # current = session.run(model.likelihood_tensor, feed_dict={inputs:X_val,labels:Y_val})
                # current = objective(model, X, Y)
                print("epoch---" + str( int(_i/iter_epochs)) + "done." , flush=True)
                if not(_i/iter_epochs==0):
                    print ("epoch time === ",time.time() - start, flush=True)
                    times.append(time.time() - start)
                start = time.time()
                model.anchor(session)
                param_dict = model.read_trainables()
                f = open("models/dgp_dropout" + str(len(model.layers)) + ".pkl","wb")
                pickle.dump(param_dict,f)
                f.close()
                if (int(_i/iter_epochs)%5==0):
                    l, a = batch_assess(model, assess_model_dgp, X_val, Y_val)
                    print('validation lik: {:.4f}, validation acc {:.4f}'.format(l, a), flush=True)
                    nlpp.append(l)
                    accuracy.append(a)
                # loss_val.append(current)
            adam()
            model.anchor(session)

    l, a = batch_assess(model, assess_model_dgp, Xs, Ys)
    print('test lik: {:.4f}, test acc {:.4f}'.format(l, a), flush=True)
    nlpp.append(l)
    accuracy.append(a)

    model.anchor(session)
            
    return np.asarray(nlpp), np.asarray(accuracy), np.asarray(times)

print("training 2 layered model")
start = time.time()
nlpp, accuracy, times = minimize(m_dgp2, 0.01, iterations, callback=None)
print ("training time === ",time.time() - start)
# np.save("loss_wconv2",loss_val)
np.save("nlpp/dgp_dropout2",nlpp)
np.save("acc/dgp_dropout2",accuracy)
np.save("time/dgp_dropout2",times)
param_dict = m_dgp2.read_trainables()
f = open("models/dgp_dropout" + str(len(m_dgp2.layers)) + ".pkl","wb")
pickle.dump(param_dict,f)
f.close()

# In[ ]:


print("training 3 layered model")
start = time.time()
nlpp, accuracy, times = minimize(m_dgp3, 0.01, iterations, callback=None)
print ("training time === ",time.time() - start)
# np.save("loss_wconv2",loss_val)
np.save("nlpp/dgp_dropout3",nlpp)
np.save("acc/dgp_dropout3",accuracy)
np.save("time/dgp_dropout3",times)
param_dict = m_dgp3.read_trainables()
f = open("models/dgp_dropout" + str(len(m_dgp3.layers)) + ".pkl","wb")
pickle.dump(param_dict,f)
f.close()

print("training 4 layered model")
start = time.time()
nlpp, accuracy, times = minimize(m_dgp4, 0.01, iterations, callback=None)
print ("training time === ",time.time() - start)
# np.save("loss_wconv2",loss_val)
np.save("nlpp/dgp_dropout4",nlpp)
np.save("acc/dgp_dropout4",accuracy)
np.save("time/dgp_dropout4",times)
param_dict = m_dgp4.read_trainables()
f = open("models/dgp_dropout" + str(len(m_dgp4.layers)) + ".pkl","wb")
pickle.dump(param_dict,f)
f.close()


sys.stdout = old_stdout

log_file.close()