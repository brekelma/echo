# Echo Noise for Exact Mutual Information Calculation

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
Experiments are specifed using the config files, which specify the network architecture and loss functions.  ```run.py``` calls ```model.py``` to parse these ```configs/``` and create / train a model.  You can also modify the tradeoff parameter ```beta``` or specify the dataset using ```'binary_mnist', 'omniglot', and 'fmnist'.``` . Analysis tools are mostly omitted, but the model loss training history is saved in a pickle file.

## Methods

```vae.json```
```infovae.json```
```vae_maf.json```
