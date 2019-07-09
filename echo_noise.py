
import keras.backend as K
import numpy as np
import tensorflow as tf 


# helper functions


# sampling with replacement, without setting batch dimension
def random_indices(n, d):
    return tf.random.uniform((n * d,), minval=0, maxval=n, dtype=tf.int32)

def gather_nd_reshape(t, indices, final_shape):
    h = tf.gather_nd(t, indices)
    return K.reshape(h, final_shape)

#
# Produce an index tensor that gives a permuted matrix of other samples in
# batch, per sample.
#
# Parameters
# ----------
# batch_size : int
#     Number of samples in the batch.
# d_max : int
#     The number of blocks, or the number of samples to generate per sample.
#
# Deps:
#   numpy
def indices_without_replacement(batch_size, d_max=-1, replace = False, pop = True):
      """Produce an index tensor that gives a permuted matrix of other samples in batch, per sample.
      Parameters
      ----------
      batch_size : int
          Number of samples in the batch.
      d_max : int
          The number of blocks, or the number of samples to generate per sample.

      replace : bool
          Assumed false, see echo_sample (and functions above) for sampling with replacement

      pop : bool
          whether current training example is excluded when constructing noise

      """
      try:
          if d_max < 0:
              d_max = batch_size + d_max
      except:
          pass
      
      
      #looping condition used below     
      cond = lambda b, i: tf.less(tf.shape(i)[0], b)
      

      i = tf.constant(0)
      batch_range = tf.range(batch_size)
      off = 0 if pop else 1
      
      if pop:
          batch_mask = tf.where(tf.equal(batch_range, i), tf.zeros_like(batch_range), tf.ones_like(batch_range))
          batch_range = tf.boolean_mask(batch_range, batch_mask)
         
      batch_shuff = tf.random.shuffle(batch_range)
      dmax_slice = batch_shuff[:d_max]
      
      dmax_range = tf.range(batch_size)[:d_max-1+off]
      dmax_enumerated = tf.concat([tf.expand_dims(dmax_range,1), tf.expand_dims(dmax_slice,1)], axis = -1)      
      inds = tf.expand_dims(dmax_enumerated,0)
      

      # Specify encoding samples to be used for Echo Noise construction 
      #     -- This loop permutes the batch for each training example, yielding :d_max indices for Echo at each data point
      def loop_call(batch, inds):
          i = tf.shape(inds)[0] 
          batch_range = tf.range(batch)
          
          if pop:
              batch_mask = tf.where(tf.equal(batch_range, i), tf.zeros_like(batch_range), tf.ones_like(batch_range))
              batch_range = tf.boolean_mask(batch_range, batch_mask)
          
          # prepare enumerated list of indices (batch, dmax, 2) 
          #     where (i,j,:) specifies 2d index to find j_th echo sample for training example i
          batch_shuff = tf.random.shuffle(batch_range)
          dmax_slice = batch_shuff[:d_max]
          dmax_range = tf.range(batch_size)[:d_max-1+off]
          dmax_enumerated = tf.concat([tf.expand_dims(dmax_range,1), tf.expand_dims(dmax_slice,1)], axis = -1)
          inds = tf.concat([inds, tf.expand_dims(dmax_enumerated, 0)], axis = 0)
          

          return [batch, inds]


      batch, inds = tf.while_loop(cond, loop_call, (batch_size, inds), 
          shape_invariants = (batch_size.get_shape(), tf.TensorShape([None,None,2])), 
          swap_memory = True, return_same_structure = True) 
      
      return inds

#
# This function implements the Echo Noise distribution specified in:
#   Exact Rate-Distortion in Autoencoders via Echo Noise
#   Brekelmans et al. 2019
#   https://arxiv.org/abs/1904.07199
#
# Parameters
# ----------
# inputs should be specified as list:
#   [ f(X), s(X) ] with s(X) in log space if calc_log = True 
# the flag plus_sx should be:
#   True if logsigmoid activation for s(X)
#   False for softplus (equivalent)
#
# Deps:
#   numpy
#   tensorflow
#   permute_neighbor_indices (above)
#   gather_nd_reshape (above)
#   random_indices (above)
def echo_sample(
    inputs,
    clip=None, d_max=100, batch=100, multiplicative=False, echo_mc = False,
    replace=False, fx_clip=None, plus_sx=True, calc_log=True,
    return_noise=False, **kwargs
    ):
    # kwargs unused

    if isinstance(inputs, list):
        fx = inputs[0]
        sx = inputs[-1]
    else:
        fx = inputs

    # TO DO : CALC_LOG currently determines both whether to do log space calculations AND whether sx is a log
 
    fx_shape = fx.get_shape()
    sx_shape = sx.get_shape()
    z_dim = K.int_shape(fx)[-1]
    batch_size = batch
    batch = K.shape(fx)[0]

    # clip is multiplied times s(x) to ensure that sum of truncated terms < machine precision 
    # clip should be calculated numerically according to App C in paper
    # M (r ^ dmax / 1-r ) < precision, SOLVE for r (clipping factor), with M = max magnitude of f(x)
    
    # calculation below is an approximation (ensuring only term d_max + 1 < precision)
    if clip is None:
        max_fx = fx_clip if fx_clip is not None else 1.0
        clip = (2**(-23)/max_fx)**(1.0/d_max)
    
    # fx_clip can be used to restrict magnitude of f(x), not used in paper
    # defaults to no clipping and M = 1 (e.g. with tanh activation for f(x))
    if fx_clip is not None: 
        fx = K.clip(fx, -fx_clip, fx_clip)

    if not calc_log:
        sx = tf.multiply(clip,sx)
        sx = tf.where(tf.abs(sx) < K.epsilon(), K.epsilon()*tf.sign(sx), sx)
    else:
        # plus_sx based on activation for sx = s(x):
        #   True for log_sigmoid
        #   False for softplus
        sx = tf.log(clip) + (-1*sx if not plus_sx else sx)

    
    if echo_mc is not None and echo_mc:    
        # use mean centered fx for noise :  performs worse
        fx = fx - K.mean(fx, axis = 0, keepdims = True)

    if replace: # replace doesn't set batch size (using permute_neighbor_indices does)
        
        sx = K.batch_flatten(sx) if len(sx_shape) > 2 else sx 
        fx = K.batch_flatten(fx) if len(fx_shape) > 2 else fx 
        
        # Sampling with replacement 
        inds = K.reshape(random_indices(batch, d_max), (-1, 1))
        select_sx = gather_nd_reshape(sx, inds, (-1, d_max, z_dim))
        select_fx = gather_nd_reshape(fx, inds, (-1, d_max, z_dim))

        if len(sx_shape)>2:
            select_sx = K.expand_dims(K.expand_dims(select_sx, 2), 2)
            sx = K.expand_dims(K.expand_dims(sx, 1),1)
        if len(fx_shape)>2:
            select_fx = K.expand_dims(K.expand_dims(select_fx, 2), 2)
            fx = K.expand_dims(K.expand_dims(fx, 1),1)

    else:
        # batch x batch x z_dim 
        # for all i, stack_sx[i, :, :] = sx
        repeat = tf.multiply(tf.ones_like(tf.expand_dims(fx, 0)), tf.ones_like(tf.expand_dims(fx, 1)))
        stack_fx = tf.multiply(fx, repeat)
        stack_sx = tf.multiply(sx, repeat)

        # select a set of dmax examples from original fx / sx for each batch entry
        inds = indices_without_replacement(batch, d_max) 
        
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

    sx = sx if not calc_log else tf.exp(sx)
    if multiplicative:
      # unused in paper, not extensively tested : log Z has Echo distribution
      output = tf.exp(fx + tf.multiply(sx, noise))
    else:
      output = fx + tf.multiply(sx, noise)
    
    return output if not return_noise else noise
    


#
# This function implements the Mutual Information penalty (via Echo Noise)
# which was specified in:
#
#   Exact Rate-Distortion in Autoencoders via Echo Noise
#   Brekelmans et al. 2019
#   https://arxiv.org/abs/1904.07199
#
# Parameters
# ----------
# inputs: tensor (or list of tensors [f(x), s(x)])
#   The sigmoid outputs from an encoder (don't include the mean outputs).
#
# clip : scales s(x) to ensure that sum of truncated terms < machine precision 
#   clip should be calculated numerically according to App C in paper
#   Solve for r:  M (r ^ dmax / 1-r ) < machine precision, with M = max magnitude of f(x)
#
# calc_log : bool whether inputs are in log space
# plus_sx : bool whether inputs measure s(x) (vs. -s(x) if parametrized with softplus, e.g.)

def echo_loss(inputs,
    clip= 0.8359, calc_log = True, plus_sx = True, multiplicative = False,
    **kwargs):
    
    if isinstance(inputs, list):
        z_mean = inputs[0]
        z_scale = inputs[-1]
    else:
        z_scale = inputs

    mi = -K.log(K.abs(clip*z_scale)+K.epsilon()) if not calc_log else -(tf.log(clip) + (z_scale if plus_sx else -z_scale))
    
    return mi


