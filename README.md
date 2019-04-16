# Echo Noise Channel for Exact Mutual Information Calculation

Code replicating the experiments in:  https://arxiv.org/abs/1904.07199
   
```
Exact Rate-Distortion in Autoencoders via Echo Noise 
Rob Brekelmans, Daniel Moyer, Aram Galstyan, Greg Ver Steeg
USC Information Sciences Institute
```


## Instructions:  
```
python run.py --config 'echo.json' --beta 1.0 --filename 'echo_example' --dataset 'binary_mnist'
```
Experiments are specifed using the config files, which specify the network architecture and loss functions.  ```run.py``` calls ```model.py``` to parse these ```configs/``` and create / train a model.  You can also modify the tradeoff parameter ```beta```, which is multiplied by the rate term, or specify the dataset using ```'binary_mnist'```, ```'omniglot'```, or ```'fmnist'.``` . Analysis tools are mostly omitted, but the model loss training history is saved in a pickle file.

## Echo

Outside of the code given here, Echo can be implemented using a similar setup to VAEs
```
z_mean = Dense(32, activation = model_utils.activations.tanh64)(h)
z_log_scale = Dense(32, activation = tf.math.log_sigmoid)(h)
z_activation = Lambda(model_utils.layers.echo_sample)([z_mean, z_log_scale])
echo_loss = Lambda(model_utils.layers.echo_loss)([z_log_scale])
```

## Comparison Methods
The following comparison methods are included in the  ```configs/``` folder.  We compare diagonal Gaussian noise encoders ('VAE') and IAF encoders, alongside several marginal approximations
