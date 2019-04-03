# Copyright 2017 Hugh Salimbeni
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np

from gpflow.params import DataHolder, Minibatch
from gpflow import autoflow, params_as_tensors, ParamList
from gpflow.models.model import Model
from gpflow.mean_functions import Identity, Linear
from gpflow.mean_functions import Zero
from gpflow import settings
from gpflow import transforms
from gpflow.conditionals import base_conditional
float_type = settings.float_type

from layers import SVGP_Layer
from utils import BroadcastingLikelihood,reparameterize

from convkernels import Conv


np.random.seed(1234)
tf.set_random_seed(1234)
class DGP_Base(Model):
    """
    The base class for Deep Gaussian process models.

    Implements a Monte-Carlo variational bound and convenience functions.

    """
    def __init__(self, X, Y, likelihood, layers,
                 minibatch_size=None,
                 num_samples=10):
        Model.__init__(self)
        self.num_samples = num_samples

        self.num_data = X.shape[0]
        if minibatch_size:
            self.X = Minibatch(X, minibatch_size, seed=0)
            self.Y = Minibatch(Y, minibatch_size, seed=0)
        else:
            self.X = DataHolder(X)
            self.Y = DataHolder(Y)

        self.likelihood = BroadcastingLikelihood(likelihood)

        self.layers = ParamList(layers)

    @params_as_tensors
    def propagate(self, X, full_cov=False, S=1, zs=None,test=0):
        
     
        sX = tf.tile(tf.expand_dims(X, 0), [S, 1, 1])

        Fs, Fmeans, Fvars = [], [], []

        F = sX
        zs = zs or [None, ] * len(self.layers)

        '''
        if test==0:
            select  = np.random.choice(X.shape[1]-1, size=int((1-self.dropout)*(X.shape[1]-1)), replace = False)
            select = np.sort(select)

        else:
            select = None
		'''

        select = None

        for layer, z in zip(self.layers, zs):
            F, Fmean, Fvar, select = layer.sample_from_conditional(F, z=z, full_cov=full_cov, test=test, select = select)

            Fs.append(F)
            Fmeans.append(Fmean)
            Fvars.append(Fvar)

        return Fs, Fmeans, Fvars

    @params_as_tensors
    def _build_predict(self, X, full_cov=False, S=1,test=0):
        Fs, Fmeans, Fvars = self.propagate(X, full_cov=full_cov, S=S,test=test)
        return Fmeans[-1], Fvars[-1]


    def E_log_p_Y(self, X, Y,test=0):
        """
        Calculate the expectation of the data log likelihood under the variational distribution
         with MC samples
        """
        Fmean, Fvar = self._build_predict(X, full_cov=False, S=self.num_samples,test=0)
        var_exp = self.likelihood.variational_expectations(Fmean, Fvar, Y)  # S, N, D
        return tf.reduce_mean(var_exp, 0)  # N, D

    @params_as_tensors
    def _build_likelihood(self):
        L = tf.reduce_sum(self.E_log_p_Y(self.X, self.Y))
        KL = tf.reduce_sum([layer.KL() for layer in self.layers])
        scale = tf.cast(self.num_data, float_type)
        scale /= tf.cast(tf.shape(self.X)[0], float_type)  # minibatch size
        return L * scale - KL

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_f(self, Xnew, num_samples):
        return self._build_predict(Xnew, full_cov=False, S=num_samples,test=1)

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_f_full_cov(self, Xnew, num_samples):
        return self._build_predict(Xnew, full_cov=True, S=num_samples,test=1)

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_all_layers(self, Xnew, num_samples):
        return self.propagate(Xnew, full_cov=False, S=num_samples, test=1)

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_all_layers_full_cov(self, Xnew, num_samples):
        return self.propagate(Xnew, full_cov=True, S=num_samples, test=1)

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_y(self, Xnew, num_samples):
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False, S=num_samples,test=1)
        return self.likelihood.predict_mean_and_var(Fmean, Fvar)

    @autoflow((float_type, [None, None]), (float_type, [None, None]), (tf.int32, []))
    def predict_density(self, Xnew, Ynew, num_samples):
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False, S=num_samples,test=1)
        l = self.likelihood.predict_density(Fmean, Fvar, Ynew)
        log_num_samples = tf.log(tf.cast(num_samples, float_type))
        return tf.reduce_logsumexp(l - log_num_samples, axis=0)

    @autoflow((float_type, [None, None]), (float_type, [None, None]), (tf.int32, []))
    def predict_density_nd_y(self, Xnew, Ynew, num_samples):
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False, S=num_samples,test=1)
        l = self.likelihood.predict_density(Fmean, Fvar, Ynew)
        log_num_samples = tf.log(tf.cast(num_samples, float_type))
        a = tf.reduce_logsumexp(l - log_num_samples, axis=0)
        b, c = self.likelihood.predict_mean_and_var(Fmean, Fvar)
        return a, b, c





class DGP(DGP_Base):
    """
    This is the Doubly-Stochastic Deep GP, with linear/identity mean functions at each layer.

    The key reference is

    ::
      @inproceedings{salimbeni2017doubly,
        title={Doubly Stochastic Variational Inference for Deep Gaussian Processes},
        author={Salimbeni, Hugh and Deisenroth, Marc},
        booktitle={NIPS},
        year={2017}
      }

    """
    def __init__(self, X, Y, Z, kernels, likelihood, 
                 num_outputs=None,
                 mean_function=Zero(),  # the final layer mean function
                 dropout =0.0,
                 **kwargs):
        Model.__init__(self)
        num_outputs = num_outputs or Y.shape[1]
        self.dropout = dropout
        # init the layers
        layers = []

        # inner layers
        X_running, Z_running = X.copy(), Z.copy()
        for kern_in, kern_out in zip(kernels[:-1], kernels[1:]):
            
            if isinstance(kern_in,Conv):
                dim_in = kern_in.basekern.input_dim
            else:
                dim_in = kern_in.input_dim
            
            dim_out = kern_out.input_dim
            
            if dim_in == dim_out:
                mf = Identity()


            else:  # stepping down, use the pca projection
                _, _, V = np.linalg.svd(X_running, full_matrices=False)
                W = V[:dim_out, :].T
                b = np.zeros(1,dtype=np.float64)
                mf = Linear(W,b)
                mf.set_trainable(False)

            mf = Zero()
            if isinstance(kern_in,Conv):
                Z_patch = np.unique(kern_in.compute_patches(Z_running).reshape(-1, kern_in.patch_len), axis=0)
                Z_patch = Z_patch[np.random.permutation((len(Z_patch)))[:Z_running.shape[0]], :]
                layers.append(svconvgp(kern_in, Z_patch,dim_out, mf, dropout))

            else:
                layers.append(SVGP_Layer(kern_in, Z_running, dim_out, mf, dropout))
                 

            if dim_in != dim_out:
                Z_running = Z_running.dot(W)
                X_running = X_running.dot(W)


        # final layer
        if isinstance(kernels[-1],Conv):
            Z_patch = np.unique(kernels[-1].compute_patches(Z_running).reshape(-1, kernels[-1].patch_len), axis=0)
            Z_patch = Z_patch[np.random.permutation((len(Z_patch)))[:Z_running.shape[0]], :]
            layers.append(svconvgp(kernels[-1], Z_patch,num_outputs, mean_function, 0.0))
        else:
            layers.append(SVGP_Layer(kernels[-1], Z_running, num_outputs, mean_function, 0.0))
        DGP_Base.__init__(self, X, Y, likelihood, layers, **kwargs)

class svconvgp(SVGP_Layer):
    def __init__(self, kern, Z, num_outputs, mean_function, dropout):
        SVGP_Layer.__init__(self, kern, Z, num_outputs, mean_function, dropout)

    def conditional(self, X, test, full_cov=False):
        """
        A multisample conditional, where X is shape (S,N,D_out), independent over samples S

        if full_cov is True
            mean is (S,N,D_out), var is (S,N,N,D_out)

        if full_cov is False
            mean and var are both (S,N,D_out)

        :param X:  The input locations (S,N,D_in)
        :param full_cov: Whether to calculate full covariance or just diagonal
        :return: mean (S,N,D_out), var (S,N,D_out or S,N,N,D_out)
        """

        def single_sample_conditional(X, full_cov=False):

            if test==0:

                select = tf.random_shuffle(tf.range(tf.shape(self.q_mu)[1]))[:tf.cast((1.0-self.dropout)*tf.cast(tf.shape(self.q_mu)[1],tf.float64),tf.int32)]
                select = tf.contrib.framework.sort(select)
                q_mu_temp = tf.gather(self.q_mu,select,axis=1)
                q_sqrt_temp = tf.gather(self.q_sqrt,select,axis=0)
                
                '''
                select  = np.random.choice((tf.convert_to_tensor(self.q_mu.shape[1])).eval()-1, size=int((1-self.dropout)*((tf.convert_to_tensor(self.q_mu.shape[1])).eval()-1)), replace = False)
                select = np.sort(select)
                q_mu_temp = np.take((self.q_mu).eval(),select,axis=1)
                q_sqrt_temp = np.take((self.q_sqrt).eval(),select,axis=0)
                transform = transforms.LowerTriangular((tf.convert_to_tensor(self.feature.Z.shape[0])).eval(), num_matrices=q_mu_temp.shape[1])
                q_sqrt_temp = Parameter(q_sqrt_temp, transform=transform)
                q_mu_temp = tf.constant(q_mu_temp)
                q_sqrt_temp = tf.constant(q_sqrt_temp)
                '''

            else:
                q_mu_temp = self.q_mu
                q_sqrt_temp = self.q_sqrt
                select = []

            self.q_mu_temp = q_mu_temp
            self.q_sqrt_temp = q_sqrt_temp

            mean, var = my_conditional(X, self.feature.Z, self.kern,
                        q_mu_temp, q_sqrt=q_sqrt_temp,
                        full_cov=full_cov, white=True)
            return mean + self.mean_function(X), var, select

        if full_cov is True:
            f = lambda a: single_sample_conditional(a, full_cov=full_cov)
            mean, var, select = tf.map_fn(f, X, dtype=(tf.float64, tf.float64))
            return tf.stack(mean), tf.stack(var), select
        else:
            S, N, D = tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2]
            X_flat = tf.reshape(X, [S * N, D])
            mean, var, select = single_sample_conditional(X_flat)
            mean = tf.reshape(mean, [S, N, -1])
            var = tf.reshape(var, [S, N, -1])
            return mean, var, select

    def sample_from_conditional(self, X, test, select, z=None, full_cov=False):
        """
        Calculates self.conditional and also draws a sample

        If z=None then the tensorflow random_normal function is used to generate the
        N(0, 1) samples, otherwise z are used for the whitened sample points

        :param X: Input locations (S,N,D_in)
        :param full_cov: Whether to compute correlations between outputs
        :param z: None, or the sampled points in whitened representation
        :return: mean (S,N,D), var (S,N,N,D or S,N,D), samples (S,N,D)
        """

        mean, var, select = self.conditional(X, full_cov=full_cov, test=test)
        if z is None:
            z = tf.random_normal(tf.shape(mean), dtype=settings.float_type)
        samples = reparameterize(mean, var, z, full_cov=full_cov)
        return samples, mean, var, select

def my_conditional(Xnew, X, kern, f, *, full_cov=False, q_sqrt=None, white=False):
        num_data = tf.shape(X)[0]  # M
        Kmm = kern.Kzz(X) + tf.eye(num_data, dtype=settings.float_type) * settings.jitter
        Kmn = kern.Kzx(X, Xnew)
        if full_cov:
            Knn = kern.K(Xnew)
        else:
            Knn = kern.Kdiag(Xnew)
        return base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white)   




