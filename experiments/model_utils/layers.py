import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Reshape, Average, Conv2D, Concatenate
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import tensorflow.keras.models
import tensorflow.keras.initializers as initializers
import tensorflow.keras.regularizers as regularizers
import tensorflow.keras.constraints as constraints
from tensorflow.keras.initializers import Constant, TruncatedNormal, RandomUniform
import numpy as np
import tensorflow as tf 
#from losses import discrim_loss, binary_crossentropy
from random import shuffle, randint
from functools import partial
import model_utils.losses as losses
import copy
import importlib 
import itertools
import tensorflow.keras.optimizers
from collections import defaultdict
import tensorflow_probability as tfp
tfd = tfp.distributions
#tfd = tf.contrib.distributions 
tfb = tfp.bijectors
#import IPython

# sampling with replacement, without setting batch dimension
def random_indices(n, d):
    return tf.random.uniform((n * d,), minval=0, maxval=n, dtype=tf.int32)

def gather_nd_reshape(t, indices, final_shape):
    h = tf.gather_nd(t, indices)
    return K.reshape(h, final_shape)


def indices_without_replacement(batch_size, d_max=-1, replace = False, pop = True, use_old = False):
      """Produce an index tensor that gives a permuted matrix of other samples in batch, per sample.
      Parameters
      ----------
      batch_size : int
          Number of samples in the batch.
      d_max : int
          The number of blocks, or the number of samples to generate per sample.
      """
      try:
          if d_max < 0:
              d_max = batch_size + d_max
      except:
          pass
      
      off = 0 if pop else 1
      
      #if tf.contrib.framework.is_tensor(batch_size) and not use_old:# replace:
      i = tf.constant(0)
    
      cond = lambda b, i: tf.less(tf.shape(i)[0], b)

      batch_range = tf.range(batch_size)
      #batch_range = tf.where(tf.equal(batch_range, i), tf.zeros_like(batch_range), batch_range)
      if pop:
          batch_mask = tf.where(tf.equal(batch_range, i), tf.zeros_like(batch_range), tf.ones_like(batch_range))
          batch_range = tf.boolean_mask(batch_range, batch_mask)
         
      batch_shuff = tf.random.shuffle(batch_range)
      dmax_slice = batch_shuff[:d_max]
      
      dmax_range = tf.range(batch_size)[:d_max-1+off]
      dmax_enumerated = tf.concat([tf.expand_dims(dmax_range,1), tf.expand_dims(dmax_slice,1)], axis = -1)      
      inds = tf.expand_dims(dmax_enumerated,0)
      

      def loop_call(batch, inds):#inputs):
          i = tf.shape(inds)[0] #tf.add(i,1)
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



# OLD FUNCTION which sets the batch size, but may be more intuitive
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

    # TO DO : CALC_LOG currently determines both whether to do log space calculations AND whether sx is a log
 
    fx_shape = fx.get_shape()
    sx_shape = sx.get_shape()
    z_dim = K.int_shape(fx)[-1]
    batch_size = batch
    batch = K.shape(fx)[0]

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
        sx = tf.multiply(clip,sx)
        sx = tf.where(tf.abs(sx) < K.epsilon(), K.epsilon()*tf.sign(sx), sx)
    #raise ValueError('calc_log=False is not supported; sx has to be log_sigmoid')
    else:
        # plus_sx based on activation for sx = s(x):
        #   True for log_sigmoid
        #   False for softplus
        sx = tf.log(clip) + (-1*sx if not plus_sx else sx)
    
    #if echo_mc is not None and echo_mc:    
      # use mean centered fx for noise
    #  fx = fx - K.mean(fx, axis = 0, keepdims = True)
        


    if replace: # replace doesn't set batch size (using permute_neighbor_indices does)
        sx = K.batch_flatten(sx) if len(sx_shape) > 2 else sx 
        fx = K.batch_flatten(fx) if len(fx_shape) > 2 else fx 
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
        
   
        # Alterntive method:  but note that permute_neighbor_indices sets the batch_size dimension != None
        # this necessitates the use of fit_generator, e.g. in training to avoid 'remainder' batches if data_size % batch > 0
        #inds = permute_neighbor_indices(batch_size, d_max, replace = replace)
        
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
    
    if multiplicative: # log z according to echo
        output = tf.exp(fx + tf.multiply(sx, noise))
    else:
        output = fx + tf.multiply(sx, noise) 

    return output if not return_noise else noise
  
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
  
  z_score = K.random_normal(shape= tf.shape(z_mean),
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
 

def clip_by_value_preserve_gradient(t, clip_value_min, clip_value_max,
                                    name=None):
    with tf.name_scope(name or 'clip_by_value_preserve_gradient'):
        t = tf.convert_to_tensor(t, name='t')
        clip_t = tf.clip_by_value(t, clip_value_min, clip_value_max)
        return t + tf.stop_gradient(clip_t - t)

def lstm_masked_template(hidden_layers,
                       shift_only=False,
                       activation=tf.nn.relu,
                       log_scale_min_clip=-5.,
                       log_scale_max_clip=3.,
                       log_scale_clip_gradient=False,
                       name=None,
                       *args,  # pylint: disable=keyword-arg-before-vararg
                       **kwargs):

    name = name or 'lstm_masked_template'
    with tf.name_scope(name):
        def _fn(x):
            """MADE parameterized via `masked_autoregressive_default_template`."""
            # TODO(b/67594795): Better support of dynamic shape.
            input_depth = tf1.dimension_value(
              tf1.TensorShape.with_rank_at_least(x.shape, 1)[-1])
            if input_depth is None:
                raise NotImplementedError(
                    'Rightmost dimension must be known prior to graph execution.')
            input_shape = (
              np.int32(tf1.TensorShape.as_list(x.shape))
              if tf1.TensorShape.is_fully_defined(x.shape) else tf.shape(x))
            print("FIRST X MASKED TEMPLATE ", x)
            #f tf1.TensorShape.rank(tf.shape(x)) == 1:
            if len(tf1.TensorShape.as_list(x.shape)) == 1:
                x = x[tf.newaxis, ...]
            for i, units in enumerate(hidden_layers):
                x = tfb.masked_dense(
                    inputs=x,
                    units=units,
                    num_blocks=input_depth,
                    exclusive=True if i == 0 else False,
                    activation=activation,
                    *args,  # pylint: disable=keyword-arg-before-vararg
                    **kwargs)
            # input depth = number of latent dims
            x = tfb.masked_dense(
              inputs=x,
              units= 2 * input_depth, #SHIFT ONLY DOESN"T MAKE SENSE for LSTM-type update
                     #(1 if shift_only else 2) * input_depth,
              num_blocks=input_depth,
              activation=None,
              *args,  # pylint: disable=keyword-arg-before-vararg
              **kwargs)
            
            #if shift_only:
            #    x = tf.reshape(x, shape=input_shape)
            #    return x, None
            x = tf.reshape(x, shape=tf.concat([input_shape, [2]], axis=0))
            # IAF lstm-type update
            shift, logit = tf.unstack(x, num=2, axis=-1)
            std = tf.nn.sigmoid(logit)
            shift = shift - tf.multiply(std, shift)
            log_scale = tf.math.log_sigmoid(logit)

            which_clip = (
              tf.clip_by_value
              if log_scale_clip_gradient else clip_by_value_preserve_gradient)
            log_scale = which_clip(log_scale, log_scale_min_clip, log_scale_max_clip)
            return shift, log_scale

    return tf1.make_template(name, _fn)

class MAF(Layer):
  def __init__(self, steps = None, layers = [640,640,640], activation = 'relu', mean_only = True, name = 'maf_chain', 
      return_both = False, add_base = False, bijector = None, maf = None, density = None, **kwargs):
    
    self.layers = tuple(layers)
    self.steps = steps if steps is not None else 1
    self.mean_only = mean_only
    self._name = name

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
      input_shape = input_shape[0]
    else:
      self.dim = input_shape[-1]
    
    if self.layers is None:
      self.layers = [self.dim*10, self.dim*10, self.dim*10]
    
    #********** HACKY: manually specified batch size... modify *********
    zeros_shape = [100, 1, 1, self.dim] if len(input_shape)==4 else [100, self.dim]

    if self.bijector is None:
    # activation defaults to relu
      maf_chain = list(itertools.chain.from_iterable([
        tfb.MaskedAutoregressiveFlow(
          shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
              hidden_layers=self.layers, shift_only=self.mean_only, activation = tf.nn.relu, name = self._name+str(i))),
              #**{"kernel_initializer": tf.ones_initializer()}))), 
        tfb.Permute(list(reversed(range(self.dim))))] #)                                                                                                                                                
            for i in range(self.steps)))


      self.bijector = tfb.Chain(maf_chain[:-1])
      
      self.maf = tfd.TransformedDistribution(
        distribution= tfd.MultivariateNormalDiag(
            loc=tf.zeros(zeros_shape), allow_nan_stats = False), # scale = tf.ones([dim]),
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
    return -K.expand_dims(self.density, 1)

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
  def __init__(self, steps = None, layers = [640,640,640], activation = 'relu', mean_only = True, name = 'iaf_chain', 
                dim = None, bijector = None, iaf = None, density = None, return_both = False, lstm = False, **kwargs):
    
    self.lstm = lstm  #lstm style update from Kingma et al.
    self.layers = layers
    self.steps = steps if steps is not None else 1
    self.mean_only = mean_only
    self._name = name
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
    return IAF(steps = self.steps, layers = self.layers, activation = self.activation, mean_only = self.mean_only, name = self._name,
               dim = self.dim, bijector = self.bijector, iaf = self.iaf, density = self.density)

  def get_config(self):
    config = {'layers': self.layers,
              'steps': self.steps,
              'mean_only': self.mean_only,
              'name': self._name,
              'activation': self.activation,
              'dim': self.dim,
              'bijector': self.bijector,
              'iaf': self.iaf,
              'density': self.density,
              'lstm': self.lstm
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
        if self.lstm:
            # **** doesn't allow mean only or consider custom activations*********
            iaf_chain = list(itertools.chain.from_iterable([
                tfb.Invert(tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=lstm_masked_template(
                        hidden_layers=self.layers,  activation = tf.nn.relu, name = self._name+str(i)))), 
                #**{"kernel_initializer": tf.ones_initializer()}))),                                                                                  
                tfb.Permute(list(reversed(range(self.dim))))] #)
                for i in range(self.steps)))
        else:
            iaf_chain = list(itertools.chain.from_iterable([
                tfb.Invert(tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                        hidden_layers=self.layers, shift_only=self.mean_only, activation = tf.nn.relu, name = self._name+str(i)))),
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
    

    self.base = self.iaf.bijector.inverse(last_samples)

    
    self.density = self.iaf.log_prob(last_samples)
    try:
        self.density = K.squeeze(K.squeeze(self.density, 1),1)
    except:
        pass
    
    return last_samples if not self.return_both else [last_samples, self.density]

  def get_density(self, x):
    return K.expand_dims(self.density, 1) 

  def compute_output_shape(self, input_shape):
    return input_shape[0] if not self.return_both else input_shape

  def get_base(self, x = None):
    print("BASE SIZE ", self.base)
    return self.base
                    
  def get_log_det_jac(self, x):
    return K.expand_dims(self.iaf.bijector.inverse_log_det_jacobian(x, 1),0)


class VampNetwork(Layer):
  def __init__(self, encoder_mu = None, encoder_var = None, layers = None,  inputs = None, input_shape = (28,28,1), activation = None, init = None, init_noise = 0.00, 
                  lr = 0.0003, arch = 'conv', **kwargs):
      # if no encoders, will train new q(z|x)
      self.layers = layers
      self.model_layers = []

      # feed encoder models (used for vae, but not iaf)
      self.encoder_mu = encoder_mu
      self.encoder_var = encoder_var
      self.init = init
      self.init_noise = init_noise

      self.inputs = inputs if inputs is not None else 500
      self.activation = activation if activation is not None else 'relu'
      #self.pseudo_inputs = []
      self.pseudo_init = init
      self.create_network = True
      self.lr = lr
      self.arch = arch # if training encoder
      self.inp_shp = input_shape

      
      super(VampNetwork, self).__init__(**kwargs)

  def build(self, input_shape):
      if isinstance(input_shape, list):
        try:
          if isinstance(input_shape[0], list):
            input_shape = input_shape[0]
        except:
          pass
        self.z_shape = input_shape[0]
        self.x_shape = input_shape[1]
      else:
        raise ValueError("Expected input shape to be a list, got: ", input_shape)

      # if isinstance(input_shape, list):
      #   try:
      #     if isinstance(input_shape[0], list):
      #       self.z_shape = input_shape[0][0]
      #       self.ldj_shape= input_shape[0][1]
      #     else:
      #       self.z_shape = input_shape[0]
      #   except:
      #     self.z_shape = input_shape[0]
      #   self.x_shape = input_shape[1]
      # else:
      #   raise ValueError("Expected input shape to be a list, got: ", input_shape)
      
      # print()
      # print("Z SHAPE ", self.z_shape)
      if len(self.pseudo_init.shape) == 1:
        self.pseudo_init = np.expand_dims(self.pseudo_init, 0)
      pseudo_dim = np.prod(np.array(self.x_shape[1:]))
      #self.pseudo_init = self.pseudo_init + np.zeros(shape = (self.inputs, pseudo_dim))
      
      #self.pseudo_init = tf.Variable(self.pseudo_init, name = 'pseudo_inputs', trainable=True, dtype = tf.float32)
      #self.pseudos = self.pseudo_init +tf.random.normal(tf.constant((self.inputs, pseudo_dim), tf.int32), mean = 0.0, stddev =self.init_noise) 
      #self.pseudo_init = self.pseudo_init + tf.random.normal(tf.constant((self.inputs, pseudo_dim), tf.int32), mean = 0.0, stddev =self.init_noise) 
      self.pseudo_init = self.pseudo_init + np.random.normal(size = (self.inputs, pseudo_dim), loc = 0.0, scale =self.init_noise) 
      self.pseudos = tf.Variable(self.pseudo_init, name = 'pseudo_inputs', trainable = True, dtype = tf.float32)
      self.pseudos = tf.identity(self.pseudos)
      
      #K.get_session().run(tf.initialize_variables([self.pseudos]))
      self.pseudo_input = Input(tensor = self.pseudos)
        #PseudoInput2(dim = pseudo_dim, init = self.pseudo_init)
      #for k in range(self.inputs):
      #  self.pseudo_inputs.append(PseudoInput(dim = pseudo_dim, init = self.pseudo_init))
      
      
      self.mu_layer = self.encoder_mu
      self.logvar_layer = self.encoder_var
      
      super(VampNetwork, self).build(input_shape) 

  def call(self, x):
    if isinstance(x, list):
      self.z = x[0]
      self.x = x[1]
    else:
      self.z = x
    
    # code to incorporate log_det_jac adjustment of encoder flow
    # if isinstance(x, list):
    #   if isinstance(x[0],list):
    #     _ = x[0][0] # density
    #     self.z = x[0][1] # iaf base
    #     # log det jac (for non-mean-only IAF... be sure to add to 'addl' in layer_args) 
    #     self.ldj = x[0][2] 
    #     self.x = x[1]
    #   else:
    #     self.z = x[0] 
    #     self.x = x[1]
    # else:
    #   self.z = x
    
    self.z = K.batch_flatten(self.z)
    

    if self.create_network:
      pdfs = []

      mu = K.batch_flatten(self.mu_layer(self.pseudo_input))
      var = K.batch_flatten(self.logvar_layer(self.pseudo_input))

      z_eval = [K.expand_dims(self.z,0), K.expand_dims(mu,1), K.expand_dims(var,1)]
      

      pdf = Lambda(losses.gaussian_pdf, arguments = {'log': True, 'negative': True})(z_eval)      
      avg = Lambda(lambda y: K.mean(y , axis = 0, keepdims = True))(pdf)
      ret = Lambda(lambda y: K.mean(y, axis = 1, keepdims = True))(avg)
      ret = Lambda(lambda y: K.sum(y, axis = -1))(ret)
      #try:
      #    return ret + tf.reduce_mean(self.ldj)
      #except:
      return ret
        

  def compute_output_shape(self, input_shape):
      return (None, 1) #input_shape[0]
