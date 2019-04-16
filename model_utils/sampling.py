import copy
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.initializers import RandomUniform





# sampling with replacement, without setting batch dimension
def random_indices(n, d):
    return tf.random.uniform((n * d,), minval=0, maxval=n, dtype=tf.int32)

def gather_nd_reshape(t, indices, final_shape):
    h = tf.gather_nd(t, indices)
    return K.reshape(h, final_shape)


def permute_neighbor_indices(batch_size, d_max=-1, replace = False, pop = True):
      """Produce an index tensor that gives a permuted matrix of other samples in batch, per sample.
      Parameters
      ----------
      batch_size : int
          Number of samples in the batch.
      d_max : int
          The number of blocks, or the number of samples to generate per sample.
      """
      if d_max < 0:
          d_max = batch_size + d_max
      inds = []
      if not replace:
        for i in range(batch_size):
          sub_batch = list(range(batch_size))
          if pop:
            # pop = False includes training sample for echo 
            # (i.e. dmax = batch instead of dmax = batch - 1)
            sub_batch.pop(i)
          np.random.shuffle(sub_batch)
          inds.append(list(enumerate(sub_batch[:d_max])))
        return inds
      else:
        for i in range(batch_size):
            inds.append(list(enumerate(np.random.choice(batch_size, size = d_max, replace = True))))
        return inds

def echo_sample(inputs, clip=None, d_max=100, batch=100, multiplicative=False, echo_mc = False,
                replace=False, fx_clip=None, plus_sx=True, calc_log=True, return_noise=False, **kwargs):
    # kwargs unused

    # inputs should be specified as list:
    #   [ f(X), s(X) ] with s(X) in log space if calc_log = True 
    # plus_sx =
    #   True if logsigmoid activation for s(X)
    #   False for softplus (equivalent)
    if isinstance(inputs, list):
        fx = inputs[0]
        sx = inputs[-1]
    else:
        fx = inputs

    # TO DO : CALC_LOG currently determines whether to do log space calculations AND whether sx is a log

    fx_shape = tf.shape(fx)
    sx_shape = tf.shape(sx)
    
    
    if clip is None:
    # clip is multiplied times s(x) to ensure that last sampled term:
    #   (clip^d_max)*f(x) < machine precision 
        max_fx = fx_clip if fx_clip is not None else 1.0
        clip = (2**(-23)/max_fx)**(1.0/d_max)
    
    # fx_clip can be used to restrict magnitude of f(x), not used in paper
    # defaults to no clipping and M = 1 (e.g. with tanh activation for f(x))
    if fx_clip is not None: 
        fx = K.clip(fx, -fx_clip, fx_clip)
    

    if not calc_log:
        raise ValueError('calc_log=False is not supported; sx has to be log_sigmoid')
    else:
        # plus_sx based on activation for sx = s(x):
        #   True for log_sigmoid
        #   False for softplus
        sx = tf.log(clip) + (-1*sx if not plus_sx else sx)
    
    if echo_mc is not None:    
      # use mean centered fx for noise
      fx = fx - K.mean(fx, axis = 0, keepdims = True)

    #batch_size = K.shape(fx)[0]
    z_dim = K.int_shape(fx)[-1]
    
    if replace: # replace doesn't set batch size (using permute_neighbor_indices does)
        inds = K.reshape(random_indices(batch, d_max), (-1, 1))
        select_sx = gather_nd_reshape(sx, inds, (-1, d_max, z_dim))
        select_fx = gather_nd_reshape(fx, inds, (-1, d_max, z_dim))

    else:
        # batch x batch x z_dim 
        # for all i, stack_sx[i, :, :] = sx
        repeat = tf.multiply(tf.ones_like(tf.expand_dims(fx, 0)), tf.ones_like(tf.expand_dims(fx, 1)))
        stack_fx = tf.multiply(fx, repeat)
        stack_sx = tf.multiply(sx, repeat)

        # select a set of dmax examples from original fx / sx for each batch entry
        inds = permute_neighbor_indices(batch, d_max, replace = replace)
        
        # note that permute_neighbor_indices sets the batch_size dimension != None
        # this necessitates the use of fit_generator, e.g. in training to avoid 'remainder' batches if data_size % batch > 0
        
        select_sx = tf.gather_nd(stack_sx, inds)
        select_fx = tf.gather_nd(stack_fx, inds)
      
    if calc_log:
        sx_echoes = tf.cumsum(select_sx, axis = 1, exclusive = True)
    else:
        sx_echoes = tf.cumprod(select_sx, axis = 1, exclusive = True)
    
    # calculates S(x0)S(x1)...S(x_l)*f(x_(l+1))
    sx_echoes = tf.exp(sx_echoes) if calc_log else sx_echoes 
    fx_sx_echoes = tf.multiply(select_fx, sx_echoes)

    # performs the sum over dmax terms to calculate noise
    noise = tf.reduce_sum(fx_sx_echoes, axis = 1)
    
    
    if multiplicative:
        # unused in paper, not extensively tested
        output = tf.exp(tf.log(fx) + (tf.log(sx) if not calc_log else sx) + tf.log(noise))
    else:
        sx = sx if not calc_log else tf.exp(sx) 
        output = fx + tf.multiply(sx, noise) 

    return output if not return_noise else noise
