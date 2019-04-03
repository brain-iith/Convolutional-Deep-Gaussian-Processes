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

from gpflow.params import Parameter, Parameterized
from gpflow.conditionals import conditional
from gpflow.features import InducingPoints
# from gpflow.kullback_leiblers import gauss_kl
from my_kl import gauss_kl
from gpflow import transforms
from gpflow import settings
from gpflow.kernels import RBF

from utils import reparameterize

np.random.seed(1234)
tf.set_random_seed(1234)

class SVGP_Layer(Parameterized):
    def __init__(self, kern, Z, num_outputs, mean_function,dropout):
        """
        A sparse variational GP layer in whitened representation. This layer holds the kernel,
        variational parameters, inducing points and mean function.

        The underlying model at inputs X is
        f = Lv + mean_function(X), where v \sim N(0, I) and LL^T = kern.K(X)

        The variational distribution over the inducing points is
        q(v) = N(q_mu, q_sqrt q_sqrt^T)

        The layer holds D_out independent GPs with the same kernel and inducing points.

        :kern: The kernel for the layer (input_dim = D_in)
        :param q_mu: mean initialization (M, D_out)
        :param q_sqrt: sqrt of variance initialization (D_out,M,M)
        :param Z: Inducing points (M, D_in)
        :param mean_function: The mean function
        :return:
        """
        Parameterized.__init__(self)
        M = Z.shape[0]

        q_mu = np.zeros((M, num_outputs))
        q_mu = q_mu.astype(np.float64,copy=False)
        self.q_mu = Parameter(q_mu)
        
        q_sqrt = np.tile(np.eye(M)[None, :, :], [num_outputs, 1, 1])
        q_sqrt = q_sqrt.astype(np.float64,copy=False)
        transform = transforms.LowerTriangular(M, num_matrices=num_outputs)
        self.q_sqrt = Parameter(q_sqrt, transform=transform)
        

        self.feature = InducingPoints(Z)
        self.kern = kern
        self.mean_function = mean_function
        self.dropout = dropout
        self.q_mu_temp = q_mu
        self.q_sqrt_temp = q_sqrt
        # self.temp_kern_shape = int((1.0-dropout)*(kern.input_dim))


    def conditional(self, X, test, select, full_cov=False):
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

        def single_sample_conditional(X, select, full_cov=False):
            
            if test==0:
                
                if select!=None:
                    Z = tf.gather(self.feature.Z, select, axis=1)
                else:
                    Z = self.feature.Z
                # temp_kern = RBF(self.temp_kern_shape, lengthscales=self.kern.lengthscales.value, variance=self.kern.variance.value)
                # temp_kern.lengthscales.set_trainable(False)
                # temp_kern.variance.set_trainable(False)
                padd = tf.zeros([tf.shape(Z)[0] , self.kern.input_dim - tf.shape(Z)[1]], dtype=tf.float64)
                Z = tf.concat([Z,padd],1)
                padd = tf.zeros([tf.shape(X)[0] , self.kern.input_dim - tf.shape(X)[1]], dtype=tf.float64)
                X = tf.concat([X,padd],1)
                select = tf.random_shuffle(tf.range(tf.shape(self.q_mu)[1]))[:tf.cast((1.0-self.dropout)*tf.cast(tf.shape(self.q_mu)[1],tf.float64),tf.int32)]
                select = tf.contrib.framework.sort(select)
                q_mu_temp = tf.gather(self.q_mu,select,axis=1)
                q_sqrt_temp = tf.gather(self.q_sqrt,select,axis=0)



                '''
                Z =  np.take((self.feature.Z).eval(),select,axis=1)
                temp_kern = gpflow.kernels.RBF(select.shape[0], lengthscales=self.kern.lengthscales, variance=self.kern.variance)
                select  = np.random.choice((tf.convert_to_tensor(self.q_mu.shape[1])).eval()-1, size=int((1-self.dropout)*float(((tf.convert_to_tensor(self.q_mu.shape[1])).eval()-1))), replace = False)
                select = np.sort(select)
                q_mu_temp = np.take((self.q_mu).eval(),select,axis=1)
                q_sqrt_temp = np.take((self.q_sqrt).eval(),select,axis=0)
                transform = transforms.LowerTriangular(Z.shape[0], num_matrices=q_mu_temp.shape[1])
                q_sqrt_temp = Parameter(q_sqrt_temp, transform=transform)
                Z = tf.constant(Z)
                q_mu_temp = tf.constant(q_mu_temp)
                q_sqrt_temp = tf.constant(q_sqrt_temp)
                '''

            else:
                Z = self.feature.Z
                # temp_kern = self.kern
                q_mu_temp = self.q_mu
                q_sqrt_temp = self.q_sqrt

            self.q_mu_temp = q_mu_temp
            self.q_sqrt_temp = q_sqrt_temp

            mean, var = conditional(X, Z, self.kern,
                        q_mu_temp, q_sqrt=q_sqrt_temp,
                        full_cov=full_cov, white=True)
            return mean + self.mean_function(X), var, select

        if full_cov is True:
            f = lambda a: single_sample_conditional(a, select=select, full_cov=full_cov)
            mean, var, select = tf.map_fn(f, X, dtype=(tf.float64, tf.float64))
            return tf.stack(mean), tf.stack(var), select
        else:
            S, N, D = tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2]
            X_flat = tf.reshape(X, [S * N, D])
            mean, var, select= single_sample_conditional(X_flat, select=select)
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

        if test==0 and select!=None:
            X = tf.gather(X,select,axis=2)

        mean, var, select = self.conditional(X, full_cov=full_cov, test=test, select = select)
        if z is None:
            z = tf.random_normal(tf.shape(mean), dtype=settings.float_type)
        samples = reparameterize(mean, var, z, full_cov=full_cov)
        return samples, mean, var, select

    def KL(self):
        """
        The KL divergence from the variational distribution to the prior

        :return: KL divergence from N(q_mu, q_sqrt) to N(0, I), independently for each GP
        """
        return gauss_kl(self.q_mu_temp, self.q_sqrt_temp)


