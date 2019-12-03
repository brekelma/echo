# Echo Noise for Exact Mutual Information Calculation

Tensorflow/Keras code accompanying:  https://arxiv.org/abs/1904.07199
Echo is a drop-in alternative for Gaussian noise that admits a simple, exact expression for mutual information.
```
@article{brekelmans2019exact,
  title={Exact Rate-Distortion in Autoencoders via Echo Noise},
  author={Brekelmans, Rob and Moyer, Daniel and Galstyan, Aram and Ver Steeg, Greg},
  journal={arXiv preprint arXiv:1904.07199},
  year={2019}
}
```


## Echo Noise

For easy inclusion in other projects, the echo noise functions are included
in one all-in-one file, ```echo_noise.py```, which can be copied to a project
and included directly, e.g.:
```python
import echo_noise
```

There are two basic functions
implemented, the noise function itself (```echo_sample```)
and the MI calculation (```echo_loss```), both of which are included in
```echo_noise.py```. Except for libaries, ```echo_noise.py``` has no other
file dependencies.

Echo noise is meant to be used similarly to the Gaussian noise in VAEs, and
was implemented with VAE implementations in mind. Assuming the inference network
provides ```z_mean``` and ```z_log_scale```, a Gaussian Encoder would look
something like:
```python
z = z_mean + tf.exp(z_log_scale) * tf.random.normal( tf.shape(z_mean) )
```
The Echo noise equivalent implemented here is:
```python
z = echo_noise.echo_sample( [z_mean, z_log_scale] )
```
Similarly, VAEs often calculate a KL divergence penalty based on
```z_mean``` and ```z_log_scale```. The Echo noise penalty, which is the
mutual information `I(x,z)`, can be computed
using:
```python
loss = ... + echo_noise.echo_loss([z_log_scale])
```
A Keras version of this might look like the following:
```python
z_mean = Dense(latent_dim, activation = model_utils.activations.tanh64)(h)
z_log_scale = Dense(latent_dim, activation = tf.math.log_sigmoid)(h)
z_activation = Lambda(echo_noise.echo_sample)([z_mean, z_log_scale])
echo_loss = Lambda(echo_noise.echo_loss)([z_log_scale])
```

These functions are also found in the experiments code, ```model_utils/layers.py``` and ```model_utils/losses.py```.


## Instructions:  
```
python run.py --config 'echo.json' --beta 1.0 --filename 'echo_example' --dataset 'binary_mnist'
```
Experiments are specifed using the config files, which specify the network architecture and loss functions.  ```run.py``` calls ```model.py``` to parse these ```configs/``` and create / train a model.  You can also modify the tradeoff parameter ```beta```, which is multiplied by the rate term, or specify the dataset using ```'binary_mnist'```, ```'omniglot'```, or ```'fmnist'.``` . Analysis tools are mostly omitted for now, but the model loss training history is saved in a pickle file.


## A note about Echo sampling and batch size:
We can choose to sample training examples with or without replacement from within the batch for constructing Echo noise.  
For sampling without replacement, we have two helper functions which shuffle index orderings for x^(l).  ```permute_neighbor_indices``` sets the output batch_size != None and is much faster.  ```indices_without_replacement``` maintains batch_size = None (e.g. for variable batch size or fitting with keras ```fit```).  Control these with ```set_batch``` option.

Also be wary of leftover batches : we choose ```d_max``` samples to construct Echo noise from within the batch, so small batches (especially without replacement) may give inaccurate noise. 



## Comparison Methods
We compare diagonal Gaussian noise encoders ('VAE') and IAF encoders, alongside several marginal approximations : standard Gaussian prior, standard Gaussian with MMD penalty (```info_vae.json``` or ```iaf_prior_mmd.json```), Masked Autoregressive Flow (MAF), and VampPrior.  All combinations can be found in the ```configs/``` folder. 


