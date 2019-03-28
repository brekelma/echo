import copy
import tensorflow as tf
import keras.backend as K
from keras.initializers import RandomUniform

def random_indices(n, d):
    return tf.random.uniform((n * d,), minval=0, maxval=n, dtype=tf.int32)


def tile_reshape(t):
    n = K.shape(t)[0]
    tdim = K.int_shape(t)[1]
    h = K.tile(t, [n, 1])
    h = K.reshape(h, (-1, n, tdim))
    return h


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
            sub_batch.pop(i)
          shuffle(sub_batch)
          inds.append(list(enumerate(sub_batch[:d_max])))
        return inds
      else:
        for i in range(batch_size):
            inds.append(list(enumerate(np.random.choice(batch_size, size = d_max, replace = True))))
        return inds

def echo_sample(inputs, clip=None, d_max=100, batch=100, multiplicative=False,
				replace=False, fx_clip=None, plus_sx=True, return_noise=False,
				noisemc=False, calc_log=True):
	# inputs should be specified as list:
	#   [ f(X), s(X) ] with s(X) in log space if calc_log = True 
	# plus_sx =
	#   True if logsigmoid activation for s(X)
	#   False for softplus (equivalent)
	if isinstance(inputs, list):
		if len(inputs) == 2:
			fx = inputs[0]
			sx = inputs[-1]
			fx_shift = None
		elif len(inputs) == 3:
			fx = inputs[0]
			fx_shift = inputs[1]
			sx = inputs[-1]
	else:
		fx = inputs
	
	if clip is None:
	# fx_clip can be used to restrict magnitude of f(x) ('mean')
	# defaults to 1 magnitude (e.g. with tanh activation for f(x))
	# clip is multiplied times s(x) to ensure that last sampled term:
	#   (clip^d_max)*f(x) < machine precision 
		max_fx = fx_clip if fx_clip is not None else 1.0
		clip = (2**(-23)/max_fx)**(1.0/d_max)
	
	# clipping can also be used to limit magnitude of f(x), not used in paper
	if fx_clip is not None: 
		fx = K.clip(fx, -fx_clip, fx_clip)
	

	if not calc_log:
		raise ValueError('calc_log=False is not supported; sx has to be log_sigmoid')
	else:
		# plus_sx based on activation for sx = s(x):
		#   True for log_sigmoid
		#   False for softplus
		sx = tf.log(clip) + (-1*sx if not plus_sx else sx)
	
	if fx_shift is not None:    
	  # echo = original fx + learnable bias + s(x)*eps                                                                                                                                        
	  orig_fx = copy.copy(fx)
	  # use mean centered fx for noise
	  fx = fx - K.mean(fx, axis = 0, keepdims = True)

	batch_size = K.shape(fx)[0]
	z_dim = K.int_shape(fx)[1]
	

	if replace:
		inds = K.reshape(random_indices(batch_size, d_max), (-1, 1))
		stack_dmax = gather_nd_reshape(sx, inds, (-1, d_max, z_dim))
		stack_zmean = gather_nd_reshape(fx, inds, (-1, d_max, z_dim))
	else:
		# NOTE : Sampling without replacement determines the batch size as currently implemented (i.e. != None)
		# this means you have to use a fit_generator to train if you'd data samples % batch != 0 (i.e. can't handle smaller batches)
		inds = permute_neighbor_indices(batch, d_max, replace = replace, pop = pop)
	    c_z_stack = tf.stack([sx for k in range(d_max)])  
	    f_z_stack = tf.stack([fx for k in range(d_max)])  

	    stack_dmax = tf.gather_nd(c_z_stack, inds)
	    stack_zmean = tf.gather_nd(f_z_stack, inds)


	if calc_log:
		noise_sx_product = tf.cumsum(stack_dmax, axis = 1, exclusive = True)
	else:
		noise_sx_product = tf.cumprod(stack_dmax, axis = 1, exclusive = True)
	
	noise_sx_product = tf.exp(noise_sx_product) if calc_log else noise_sx_product
	# calculates S(x0)S(x1)...S(x_l)*f(x_(l+1))
	noise_times_sample = tf.multiply(stack_zmean, noise_sx_product)
   
	# performs the sum over dmax terms to calculate noise
	noise_tensor = tf.reduce_sum(noise_times_sample, axis = 1)
	
	if noisemc:
		noise_tensor -= tf.reduce_mean(noise_tensor, axis=0)
		# 0 mean noise : ends up being 1 x m
	
	if multiplicative:
		# unused in paper
		noisy_encoding = tf.multiply(fx, tf.multiply(sx, noise_tensor))
	else:
		sx = sx if not calc_log else tf.exp(sx) 
		if fx_shift is not None:
			while len(K.int_shape(fx_shift)) < len(K.int_shape(orig_fx)):
				fx_shift = K.expand_dims(fx_shift,0)

			noisy_encoder = fx_shift + orig_fx + tf.multiply(sx, noise_tensor)
		else:
			noisy_encoding = fx + tf.multiply(sx, noise_tensor) 


	return noisy_encoding if not return_noise else noise_tensor


class ShiftConstant(Layer):
  def __init__(self,  latents = None, # only necessary if you're calling on a layer with different last dimension than desired for latent space
  					  init = 1.0, # width of uniform mean-shift initialization
  					  scale = 1, # could speed up gradient descent by taking bigger steps
  					  activation = None, **kwargs):
	  self.latents = latents
	  self.trainable = True
	  self.beta = init
	  self.scale = scale
	  super(ShiftConstant, self).__init__(**kwargs)

  def build(self, input_shape):
	  if self.latents is None:
		self.latents = input_shape[-1]
	
	  self.betas = self.add_weight(name='beta',
									  shape = (self.latents,),
									  initializer= RandomUniform(minval=-self.init, maxval = self.init),
									  trainable= True)
	  super(ShiftConstant, self).build(input_shape)

  def call(self, x):    
	return self.betas*self.scale


  def compute_output_shape(self, input_shape):
	  return (self.latents,)
