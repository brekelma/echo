import sys
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import objectives
import keras.losses
import keras.models as km
from keras.layers import Lambda, Concatenate, average, concatenate, add
from keras.callbacks import Callback, TensorBoard

from functools import partial
import tensorflow_probability as tfp
tfd = tfp.distributions
#tfd = tf.contrib.distributions                                                                                                             
tfb = tfp.bijectors

EPS = K.epsilon()

# silly placeholders for keras training, i.e. dim_sum([x_true, already_calculated_loss])
def dim_sum(true, tensor, keepdims = False):
    return K.sum(tensor, axis = -1, keepdims = keepdims)

def dim_sum_one(tensor, keepdims = False):
   
    return K.sum(tensor, axis = -1, keepdims = keepdims)

def mean_one(tensor, dummy = None, keepdims = False):
    if isinstance(tensor, list):
        tensor = tensor[0]
    return K.mean(tensor, axis = 0, keepdims = keepdims)

def identity(true, tensor):
    return tensor

def identity_one(tensor):
    return tensor

def mean_mean(tensor):
    return K.mean(K.mean(tensor, axis = -1), axis = 0)

def loss_val(tensor):
    return K.mean(K.sum(tensor, axis = -1), axis = 0)

def sum_all(tensor):
    return K.sum(tensor)

def logsumexp(x, axis = -1, keepdims = False):
    #return tf.reduce_logsumexp(mx, axis = axis)
    m = K.max(x, axis=axis, keepdims = True) # keep dims for broadcasting
    return m + K.log(K.sum(K.exp(x - m), axis=axis, keepdims=keepdims)) + K.epsilon()

def logmeanexp(x, axis = -1, keepdims = False):
    m = K.max(x, axis=axis, keepdims= True)
    return m + K.log(K.mean(K.exp(x - m), axis=axis, keepdims=keepdims)) + K.epsilon()

def compute_kernel(x, y, bw = None):
    x = tf.squeeze(x)
    y = tf.squeeze(y)
    dim = tf.shape(x)[-1]
    if bw is None:
        bw = tf.sqrt(tf.cast(dim*dim, tf.float32)/2.0) # infovae paper default
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]

    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_sum(tf.square(tiled_x - tiled_y), axis=2) / (tf.cast(2*(bw*bw), tf.float32)))
    
    #InfoVAE paper implementation
    #return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2)/(tf.cast(dim, tf.float32)))


def mmd_loss(inputs, kernel = 'gaussian', gaussian = True, d=500, gamma = 1.0, bw = None):
    if not isinstance(inputs, list):
        q = inputs
        p = tfd.Normal(loc=tf.zeros_like(q), scale=tf.ones_like(q)).sample()
    elif len(inputs) == 1:
        q = inputs[0]
        p = tfd.Normal(loc=tf.zeros_like(q), scale=tf.ones_like(q)).sample()
    else:
        q, p = inputs
    
    if kernel == 'gaussian':
        q_kernel = compute_kernel(q, q, bw = bw)
        p_kernel = compute_kernel(p, p, bw = bw)
        qp_kernel = compute_kernel(q, p, bw = bw)
        mmd = tf.reduce_mean(q_kernel) + tf.reduce_mean(p_kernel) - 2 * tf.reduce_mean(qp_kernel)
        return tf.expand_dims(tf.expand_dims(mmd, 0), 1)
    else: 
        W = tf.random_normal((K.int_shape(q)[-1], d))
        phi_Wq = tf.sqrt(2/gamma) * tf.matmul(q,W) + tf.transpose(2*np.pi*tf.random_uniform((d,1)))
        phi_Wq = tf.sqrt(2/d) * tf.cos(phi_Wq)
        phi_Wp = tf.sqrt(2/gamma) * tf.matmul(p,W) + tf.transpose(2*np.pi*tf.random_uniform((d,1)))
        phi_Wp = tf.sqrt(2/d) * tf.cos(phi_Wp)
        
        mmd = K.mean((phi_Wq - phi_Wp), axis = 0, keepdims = True)**2
        return K.sum(mmd, axis = -1, keepdims = True) 

def gaussian_pdf_np(eval_pt, mu = 0.0, logvar = 0.0, log = False):
    if eval_pt is None:
        eval_pt = mu
    
    var = np.exp(logvar)
    log_pdf = -.5*np.log(2*np.pi) - (logvar / 2.0)- (eval_pt - mu)**2 / (2 * var) 

    return np.exp(log_pdf) if not log else log_pdf


def gaussian_pdf(inputs, log = False, negative = True):
    if isinstance(inputs, list):
        eval_pt = inputs[0]
        try:
            mu = inputs[1]
            while len(K.int_shape(eval_pt)) < len(K.int_shape(mu)):
                eval_pt = K.expand_dims(eval_pt, 1)
        except:
            pass
        try:
            mu = inputs[1]
        except:
            mu = 0.0
        try:
            logvar = inputs[2]
        except:
            logvar = 0.0
    else:
        eval_pt = inputs
        mu = 0.0
        logvar = 0.0

    var = K.exp(logvar)
    log_pdf = -.5*K.log(2*np.pi) - (logvar / 2.0)- (eval_pt - mu)**2 / (2 * var) 

    res= K.exp(log_pdf) if not log else log_pdf
    return -res if negative else res


def echo_loss(inputs, clip= 0.8359, calc_log = True, plus_sx = True, multiplicative = False, **kwargs):                                                                                                                                                  
    if isinstance(inputs, list):                                                                                                                                          
        z_mean = inputs[0]
        z_scale = inputs[-1]                                                                                                                                 
    else:
        z_scale = inputs
   
    # calc_log indicates whether z_scale is already in log terms
    print("*"*50)
    print("INPUTS FOR ECHO NOISE ", inputs)
    print("*"*50)
    mi = -K.log(K.abs(clip*z_scale)+K.epsilon()) if not calc_log else -(tf.log(clip) + (z_scale if plus_sx else -z_scale))
    
    return mi


def gaussian_kl(inputs):
    [mu1, logvar1, mu2, logvar2] = inputs
    return .5*logvar2-.5*logvar1 + tf.divide(K.exp(logvar1) + (mu1 - mu2)**2, 2*K.exp(logvar2)+K.epsilon()) - .5
    #return K.sum(.5*logvar2-.5*logvar1 + tf.divide(K.exp(logvar1) + (mu1 - mu2)**2,(2*K.exp(logvar2)+K.epsilon())) - .5, axis = -1)


def gaussian_prior_kl(inputs):
    [mu1, logvar1] = inputs
    mu2 = K.variable(0.0)
    logvar2 = K.variable(0.0)
    return gaussian_kl([mu1, logvar1, mu2, logvar2])

def binary_kl(inputs):
    [mu1, mu2] = inputs
    mu1 = K.clip(mu1, K.epsilon(), 1-K.epsilon())
    mu2 = K.clip(mu2, K.epsilon(), 1-K.epsilon())
    return tf.multiply(mu1, K.log(mu1)) + tf.multiply(1-mu1, K.log(1-mu1))- tf.multiply(mu1, K.log(mu2)) - tf.multiply(1-mu1, K.log(1-mu2))

def categorical_crossentropy(inputs):
    print("Categorical cross entropy inputs : ", inputs)
    if len(inputs) == 2:
        [mu1, mu2] = inputs
        if isinstance(mu2, list):
            return average([K.categorical_crossentropy(mu1, pred) for pred in mu2])
        else:
            return K.categorical_crossentropy(mu1, mu2)
    else:
        true = inputs[0]
        return average([K.categorical_crossentropy(true, inputs[pred]) for pred in range(1, len(inputs))])


def binary_crossentropy(inputs):
    if len(inputs) == 2:
        [mu1, mu2] = inputs
        if isinstance(mu2, list):
            return average([K.binary_crossentropy(mu1, pred) for pred in mu2])
        else:
            return K.binary_crossentropy(mu1, mu2)
    else:
        raise Exception("BINARY CROSS ENTROPY HAS MORE THAN 2 ARGUMENTS.  ENSURE THIS IS DESIRED")
        true = inputs[0]
        return average([K.binary_crossentropy(true, inputs[pred]) for pred in range(1, len(inputs))])

def keras_bce(x_true, x_pred):
    return K.sum(binary_crossentropy([x_true, x_pred]), axis = -1)



def mean_squared_error(inputs):
    if len(inputs) == 2:
        [mu1, mu2] = inputs
        if isinstance(mu2, list):
            return average([mse(mu1, pred) for pred in mu2])
        else:
            return mse(mu1, mu2)
    else:
        true = inputs[0]
        return average([mse(true, inputs[pred]) for pred in range(1, len(inputs))])

def mse(a, b):
    return (a-b)**2


def gaussian_neg_ent(inputs):#logvar = 0.0):
    '''calculates (conditional) entropy per sample, with logvar summed over dimensions '''
    if not isinstance(inputs, list) or len(inputs) == 1:
        logvar = inputs
    else:
        [mu, logvar] = inputs
    return -.5*(K.int_shape(logvar)[-1]*np.log(2*np.pi*np.exp(1))+K.sum(logvar, axis = -1, keepdims = True))

def lognormal_entropy(inputs):#mean, logvar):
    [mu, logvar] = inputs
    return K.sum(mu, axis = -1) + gaussian_entropy(logvar)
