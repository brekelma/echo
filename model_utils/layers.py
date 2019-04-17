import keras.backend as K
from keras.engine.topology import Layer
from keras.layers import Input, Lambda, merge, Dense, Flatten, Reshape, Average, Conv2D, Concatenate
from keras import activations
from keras.optimizers import Adam
from keras.models import Model
import keras.models
import keras.initializers as initializers
import keras.regularizers as regularizers
import keras.constraints as constraints
from keras.initializers import Constant, TruncatedNormal, RandomUniform
import numpy as np
import tensorflow as tf 
#from losses import discrim_loss, binary_crossentropy
from random import shuffle, randint
from functools import partial
import model_utils.losses as losses
import copy
import importlib 
import itertools
import keras.optimizers
from collections import defaultdict
import tensorflow_probability as tfp
tfd = tfp.distributions
#tfd = tf.contrib.distributions 
tfb = tfp.bijectors
#import IPython


def list_id(x, name = None):
  if isinstance(x, list):
    if isinstance(x[0], list):
      return x[0]
    else:
      return x
  else:
    return x
  

def shuffle_batch(x, numpy = True, batch_size = 100):
  # only works for 2d / flattened tensors
  if isinstance(x, np.ndarray):
    x = K.variable(x)
    x = K.Flatten()(x)
  batch = x.get_shape().as_list()[0] #K.int_shape(x)[0] 
  m = x.get_shape().as_list()[-1] #K.int_shape(x)[-1]
  
  if batch is None:
    batch = batch_size
  
  def shuffle_with_return(y):
    zz = copy(y)
    for i in range(zz.shape[-1]):
      np.random.shuffle(zz[:,i])
    return zz
  
  perm_matrix = np.array([[[row, j] for row in shuffle_with_return(np.arange(batch))] for j in range(m)])
  
  return tf.transpose(tf.gather_nd(x, perm_matrix)) 

def vae_sample(inputs, std = 1.0, return_noise = False, try_mvn = False):
  # standard reparametrization trick: N(0,1) => N(mu(x), sigma(x))
  z_mean, z_noise = inputs
  
  try:
    z_score = K.random_normal(shape=(z_mean._keras_shape[-1],),
                                mean=0.,
                                stddev=std)
  except:
    try:
      z_score = K.random_normal(shape=(z_mean.get_shape().as_list()[-1],),
                                mean=0.,
                                stddev=std)
    except:
      z_score = K.random_normal(shape=(z_mean.shape[-1],),
                                mean=0.,
                                stddev=std)
  return z_mean + K.exp(z_noise / 2) * z_score if not return_noise else K.expand_dims(z_score, 0)
    

def vae_np(inputs, std = 1.0):
  z_mean, z_noise = inputs
  #if not hasattr(z_mean, '_keras_shape'):
  #  z_mean = K.variable(z_mean)
  z_score = np.random.normal(size = z_mean.shape)
  return z_mean + np.exp(z_noise / 2) * z_score

def ido_sample(inputs):
  # reparametrization trick in log normal space (i.e. multiplicative noise)
  z_mean, z_noise = inputs
  std = 1.0
  z_score = K.random_normal(shape=(z_mean._keras_shape[-1],),
                                  mean=0.,
                                  stddev=std)
    
  return K.exp(z_mean + K.exp(z_noise / 2) * z_score)
 

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


class MAF(Layer):
  def __init__(self, steps = None, layers = None, activation = 'relu', mean_only = True, name = 'maf_chain', 
      return_both = False, add_base = False, bijector = None, maf = None, density = None, **kwargs):
    
    self.layers = tuple(layers)
    self.steps = steps if steps is not None else 1
    self.mean_only = mean_only
    self.name = name
    try:
      mod = importlib.import_module('keras.activations')
      self.activation = getattr(mod, self.activation)
    except:
      try:
        mod = importlib.import_module(self.activation)
        self.activation = getattr(mod, self.activation.split(".")[-1])
      except:
        self.activation = activation

    self.add_base = add_base
    self.bijector = bijector
    self.maf = maf
    self.density = density
    self.return_both = return_both
    super(MAF, self).__init__(**kwargs)

  def build(self, input_shape):
    # input shape is list of [z_mean, z_std]                                                                                                                                                       


    if isinstance(input_shape, list):
      self.dim = input_shape[0][-1]
    else:
      self.dim = input_shape[-1]

    if self.layers is None:
      self.layers = [self.dim, self.dim, self.dim]

    if self.bijector is None:
      maf_chain = list(itertools.chain.from_iterable([
        tfb.MaskedAutoregressiveFlow(
          shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
              hidden_layers=self.layers, shift_only=self.mean_only, name = self.name+str(i))),
              #**{"kernel_initializer": tf.ones_initializer()}))),                                                                                                                                      
        tfb.Permute(list(reversed(range(self.dim))))] #)                                                                                                                                                
            for i in range(self.steps)))


      self.bijector = tfb.Chain(maf_chain[:-1])
      
      self.maf = tfd.TransformedDistribution(
        distribution= tfd.MultivariateNormalDiag(
        loc=tf.zeros([self.dim]), allow_nan_stats = False), # scale = tf.ones([dim]),
        bijector= self.bijector, 
        name = 'maf_chain')
      

    self.built = True

  def call(self, inputs):
    if isinstance(inputs, list):
      inputs = inputs[-1]

    self.last_x = inputs                                                                                                                                                                                                

    def positive_log_prob(dist, x):
      #, event_ndims=0                                                                                                                                                                                      
      return (dist.bijector.inverse_log_det_jacobian(x, 1) +
              dist.distribution.log_prob(dist.bijector.inverse(x)))

    def negative_log_prob(dist, x):
      #, event_ndims=0                                                                                                                                                                                      
      return -(dist.bijector.inverse_log_det_jacobian(x, 1) +
              dist.distribution.log_prob(dist.bijector.inverse(x)))
    
    self.density = self.maf.log_prob(inputs)

    self.base = self.maf.bijector.forward(inputs)

    return -self.density if not self.return_both else [-self.density, self.base]

  def get_density(self, x):
    return K.expand_dims(self.density, 1)

  def get_base_output(self, x = None):
    return self.base
    
  def compute_output_shape(self, input_shape):
    if not self.return_both:
      return input_shape[0] if isinstance(input_shape[0], list) else input_shape
    else:
      return [input_shape, input_shape]

  def get_log_det_jac(self, x):
    return self.maf.bijector.inverse_log_det_jacobian(x, 1)


class IAF(Layer):
  def __init__(self, steps = None, layers = None, activation = 'relu', mean_only = True, name = 'iaf_chain', 
                dim = None, bijector = None, iaf = None, density = None, return_both = False, **kwargs):
    
    self.layers = layers
    self.steps = steps if steps is not None else 1
    self.mean_only = mean_only
    self.name = name
    try:
      mod = importlib.import_module('keras.activations')
      self.activation = getattr(mod, self.activation)
    except:
      try:
        mod = importlib.import_module(self.activation)
        self.activation = getattr(mod, self.activation.split(".")[-1])
      except:
        self.activation = activation
    self.dim = dim
    self.bijector = bijector
    self.iaf = iaf
    self.density = density
    self.return_both = return_both

    super(IAF, self).__init__(**kwargs)

  def __deepcopy__(self):
    return IAF(steps = self.steps, layers = self.layers, activation = self.activation, mean_only = self.mean_only, name = self.name,
               dim = self.dim, bijector = self.bijector, iaf = self.iaf, density = self.density)

  def get_config(self):
    config = {'layers': self.layers,
              'steps': self.steps,
              'mean_only': self.mean_only,
              'name': self.name,
              'activation': self.activation,
              'dim': self.dim,
              'bijector': self.bijector,
              'iaf': self.iaf,
              'density': self.density
    }
    base_config = super(IAF, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    # input shape is list of [z_mean, z_std]
    if isinstance(input_shape, list):
      self.dim = input_shape[0][-1]
    else:
      self.dim = input_shape[-1]

    if self.layers is None:
      self.layers = [self.dim, self.dim, self.dim]
    
    if self.bijector is None:
      iaf_chain = list(itertools.chain.from_iterable([
        tfb.Invert(tfb.MaskedAutoregressiveFlow(
          shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
              hidden_layers=self.layers, shift_only=self.mean_only, name = self.name+str(i)))),
              #**{"kernel_initializer": tf.ones_initializer()}))),
        tfb.Permute(list(reversed(range(self.dim))))] #)
            for i in range(self.steps)))

      self.bijector = tfb.Chain(iaf_chain[:-1]) 

      
    self.built = True
    
  def call(self, inputs):
    z_mean = inputs[0]
    z_logvar = inputs[-1]
    
    self.iaf =  tfd.TransformedDistribution(
        distribution = tfd.MultivariateNormalDiag(loc = z_mean,  
        scale_diag = tf.exp(.5*z_logvar)),
    bijector = self.bijector,
      name = 'iaf_chain')
    
    last_samples = self.iaf.sample()
    
    #self.density = self.iaf.log_prob(last_samples)
    
    #try:
    self.density = K.squeeze(K.squeeze(self.iaf.log_prob(last_samples), 1),1)
    #except:
    #  pass
    
    return (last_samples) if not self.return_both else [last_samples, self.density]

  def get_density(self, x):
    return K.expand_dims(self.density, 1) 

  def compute_output_shape(self, input_shape):
    return input_shape[0] if not self.return_both else input_shape

  def get_log_det_jac(self, x):
    return self.iaf.inverse_log_det_jacobian(x, 0)





class PseudoInput(Layer):
  def __init__(self, dim = 784, init = None, **kwargs):
      self.shape = dim
      self.init_noise = 0.01
      self.init = init + np.random.normal(scale=self.init_noise, size = init.shape) if init is not None else None
      self.initialize = True
      self.trainable = True
      super(PseudoInput, self).__init__(**kwargs)

  def build(self, input_shape):
      self.dim = input_shape[-1]
      self.full_shape = input_shape
      # random init (but overridden if self.init is given, which feeds starting values e.g. input mean) 
      self.pixels = self.add_weight(name='pseudos', 
                                      shape = (self.shape,),
                                      initializer= TruncatedNormal(mean=0.5, stddev = 0.25),
                                      trainable= True)
      super(PseudoInput, self).build(input_shape) 

  def call(self, x):
    if self.initialize and self.init is not None:
      self.set_weights([self.init])
      self.initialize = False
 
    return K.expand_dims(self.pixels,0) #tf.multiply(K.ones_like(x), self.pixels)


  def get_inputs(self):
      return self.get_weights()[0][0]


  def compute_output_shape(self, input_shape):
      return input_shape



class VampNetwork(Layer):
  def __init__(self, encoder_mu = None, encoder_var = None, layers = None, inputs = None, input_shape = (28,28,1), activation = None, init = None, lr = 0.0003, arch = 'conv', **kwargs):
      # if no encoders, will train new q(z|x)
      self.layers = layers
      self.model_layers = []

      # feed encoder models (used for vae, but not iaf)
      self.encoder_mu = encoder_mu
      self.encoder_var = encoder_var

      self.inputs = inputs if inputs is not None else 500
      self.activation = activation if activation is not None else 'relu'
      self.pseudo_inputs = []
      self.pseudo_init = None
      self.create_network = True
      self.lr = lr
      self.arch = arch # if training encoder
      self.inp_shp = input_shape

      
      super(VampNetwork, self).__init__(**kwargs)

  def build(self, input_shape):
      if isinstance(input_shape, list):
        self.z_shape = input_shape[0]
        self.x_shape = input_shape[1]
      else:
        raise ValueError("Exprected input shape to be a list, got: ", input_shape)
      
      for k in range(self.inputs):
        self.pseudo_inputs.append(PseudoInput(dim = self.x_shape[-1], init = self.pseudo_init))
      
      if self.encoder_mu is None:
        for i in range(len(self.layers)):
          layer_size = self.layers[i]
          if self.arch == 'dense':
            self.model_layers.append(Dense(layer_size, activation = self.activation, name = 'vamp_'+str(i)))
          elif self.arch == 'alemi' or self.arch == 'conv':
            sizes = {0:5, 1:5, 2:5, 3:5, 4:7}
            stride = {0:1, 1:2, 2:1, 3:2, 4:1}
            self.model_layers.append(Conv2D(layer_size, sizes[i], activation = self.activation, strides= stride[i], padding = 'same' if i < len(self.layers)-1 else 'valid',
                                         name = 'vamp_'+str(i)))
        self.mu_layer = Dense(self.z_shape[-1], activation = 'linear', name = 'z_mean_vamp')
        self.logvar_layer = Dense(self.z_shape[-1], activation = 'linear', name = 'z_logvar_vamp')
      else:
        # use already trained encoder
        self.mu_layer = self.encoder_mu
        self.logvar_layer = self.encoder_var
      
      super(VampNetwork, self).build(input_shape) 

  def call(self, x):
    print("Input VAMP ", x)
    if isinstance(x, list):
      self.z = x[0]
      self.x = x[1]
    else:
      self.z = x
    
    self.z = K.batch_flatten(self.z)

    if self.create_network:
      pdfs = []
      #inp = Input(tensor = self.x)#self.x_shape[1:])
      #print("Inp1 ", inp)
      #inp2 = Input(tensor = self.z) #shape = self.z_shape[1:])
      #print("Inp2 ", inp2)
      self.pseudos=[]
      for i in range(len(self.pseudo_inputs)):
        pseudo = self.pseudo_inputs[i](self.x)
        if self.arch == 'conv':
          pseudo = Reshape(self.inp_shp)(pseudo)
        self.pseudos.append(pseudo)

      self.pseudos = Concatenate(axis = 0)(self.pseudos)

      if self.encoder_mu is None:
        h = self.pseudos
        for j in range(len(self.model_layers)):
            h = self.model_layers[j](h)
        mu = K.batch_flatten(self.mu_layer(h))
        var = K.batch_flatten(self.logvar_layer(h))
      else:
        mu = K.batch_flatten(self.mu_layer(self.pseudos))
        var = K.batch_flatten(self.logvar_layer(self.pseudos))


      z_eval = [K.expand_dims(self.z,0), K.expand_dims(mu,1), K.expand_dims(var,1)]
      
      # eval on 1 x batch x z_dim vs. pseudos x 1 x z_dim, then average over pseudos
      pdf = Lambda(losses.gaussian_pdf, arguments = {'log': True, 'negative': True})(z_eval)      
      avg = Lambda(lambda x: K.mean(x ,axis =0, keepdims = True))(pdf)
      
      return avg
        


  def compute_output_shape(self, input_shape):
      return input_shape[0]
