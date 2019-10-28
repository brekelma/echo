# Echo Noise experiments

Tensorflow/Keras code replicating the experiments in:  https://arxiv.org/abs/1904.07199
   
```
@article{brekelmans2019exact,
  title={Exact Rate-Distortion in Autoencoders via Echo Noise},
  author={Brekelmans, Rob and Moyer, Daniel and Galstyan, Aram and Ver Steeg, Greg},
  journal={arXiv preprint arXiv:1904.07199},
  year={2019}
}
```

## Rate-Distortion Experiments
```
python run.py --config 'echo.json' --beta 1.0 --filename 'echo_example' --dataset 'binary_mnist'
```
Experiments are specifed using the config files, which specify the network architecture and loss functions.  ```run.py``` calls ```model.py``` to parse these ```configs/``` and create /\
 train a model.  You can also modify the tradeoff parameter ```beta```, which is multiplied by the rate term, or specify the dataset using ```'binary_mnist'```, ```'omniglot'```, or ```\
'fmnist'.```  Analysis tools are mostly omitted for now, but the model loss training history is saved in a pickle file.

We compare diagonal Gaussian noise encoders ('VAE') and IAF encoders, alongside several marginal approximations : standard Gaussian prior, standard Gaussian with MMD penalty (```info_vae.json``` or ```iaf_prior_mmd.json```), Masked Autoregressive Flow (MAF), and VampPrior.  All combinations can be found in the ```configs/``` folder. 


Model construction generally works by parsing the list of layers with their respective arguments (```model_utils/layer_args.py```), and feeding their outputs to losses (```model_utils/loss_args.py```) as specified in ```configs/*.json```.   It is admittedly confusing to track all of the arguments and inputs, so try printing model summaries or inputs to the respective layers (```model_utils/layers.py```) or functions (```model_utils/losses.py```).  Consider bash scripts to loop over betas or runs.

## Disentanglement Experiments
Code for the disentanglement experiments is located in ```disentanglement.py``` and should be run using command line arguments and/or bash scripts.  

Downsampling of the dataset to induce dependent latent factors is done by specifying the dataset as ```ds_corr``` instead of ```dsprites```, with code in ```dataset.py```.  We partition the values of each ground truth factor into ```n_bins``` contiguous regions.  For each pair of bins in different factors, we remove all training examples taking values in these ranges with probability ```remove_prob```.  For example, excluding shape = square, rotation < $\frac{pi}/2$ removes this combination in all x-y positions, scales, meaning that the size of the resulting dataset is less than ```remove_prob * n_train```


Note, code from the Google research disentanglement library https://github.com/google-research/disentanglement_lib/blob/master/LICENSE was used for calculating FactorVAE and MIG scores.  Thanks to them and please inform of any issues.
