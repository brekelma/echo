import numpy as np
import keras.backend as K
import importlib
import tensorflow as tf
import model_utils.losses as l
import model_utils.layers as layers
#import model
from keras.layers import Lambda
import keras.models

#from model import RECON_LOSSES

RECON_LOSSES = ['bce', 'mse', 'binary_crossentropy', 'mean_square_error', 'mean_squared_error', 'iwae']
LOSS_WEIGHTS = 'loss_weights' # path to loss weights module

class Loss(object):

    def __init__(self, beta = None, **kwargs):
        args = {
            'type': 'vae',
            'layer': -1,
            'encoder': True,
            'weight': 1, # float or string specifying function
            'output': -1, # output is data layer 
            'output_density': False,
            'name': None,
            'recon': None,
            'callback': False,
            'noise': False,
            'from_layer': [], # "stats", "recon", in order of list to be returned (stats = z_mean, z_var ... recon = x_true, x_pred)
            'from_output': [], 
            'loss_kwargs': {}
        }
        args.update(kwargs) 
        for key in args.keys():
            setattr(self, key, args[key])

        self.beta = beta 
        self.name = self.type if self.name is None else self.type+self.name

    def set_beta(self, beta):
        self.beta = beta


    def get_dict(self):
        return {'type': self.type, 'layer': self.layer, 'encoder': self.encoder, 'weight':self.weight,
                'output': self.output, 'recon': self.recon, 'from_layer': self.from_layer, 'from_output': self.from_output}
     
    def make_function(self):
        # interpret type, taking stats and/or recon as inputs 
        function = self._type_to_function()
        return function

    def describe_inputs(self):
        try:
            return self.from_layer, self.from_output
        except:
            _ = self._type_to_function()   # type to function sets from_layer/output to set inputs 
            return self.from_layer, self.from_output

    def get_loss_weight(self):
        return self._get_loss_weight()

    def _type_to_function(self):
        # naming to help handling in model.py
        name_suffix = '_'+('recon' if self.type in RECON_LOSSES else ('reg' if self.weight != 0 else ''))+('_'+str(self.layer) if self.layer != -1 else '')
        name_suffix = name_suffix + '_' + self.name if self.name is not None else name_suffix
        
        self.from_addl = []

        # *** RECON *** 
        if self.type in RECON_LOSSES:
            if self.type == 'bce' or self.type == 'binary_crossentropy' or self.type == 'binary_cross_entropy':
                self.from_output = ['act']    
                return Lambda(l.binary_crossentropy, name = 'bce'+name_suffix)
            elif self.type == 'mse' or self.type == 'mean_square_error' or self.type == 'mean_squared_error':
                self.from_output = ['act']    
                return Lambda(l.mean_squared_error, name = 'mse'+name_suffix)

        # *** VAE ***
        elif self.type == 'vae':
            self.from_layer = ['stat']
            return Lambda(l.gaussian_prior_kl, name = 'vae'+name_suffix)

        # *** INFO DROPOUT ***
        elif self.type == 'ido' or self.type == 'info_dropout':        
            self.from_layer = ['stat']
            return Lambda(l.gaussian_prior_kl, name = 'ido'+name_suffix)

        elif self.type in ['gaussian_pdf', 'gaussian_logpdf']:
            self.from_layer = ['act']
            return Lambda(l.gaussian_pdf, arguments = {'negative': True, 'log': True}, name = 'gaussian_prior_'+name_suffix)

        elif self.type == 'gaussian_ent':
            self.from_layer = ['stat']
            return Lambda(l.gaussian_neg_ent, name = 'vae_ent'+name_suffix)


        elif self.type == "echo":# in self.type:
            self.from_layer = ['stat']
            return Lambda(l.echo_loss, arguments = self.loss_kwargs, name = 'mi_echo'+name_suffix)

        elif self.type == 'mmd':
            self.from_layer = ['act']
            self.from_output = [] # for prior MMD
            return Lambda(l.mmd_loss, arguments = self.loss_kwargs, name = 'mmd'+name_suffix)
        
        elif self.type == 'mmd_density':
            self.from_layer = ['act']
            self.from_output = ['addl'] # for MMD with density estimator output
            return Lambda(l.mmd_loss, arguments = self.loss_kwargs, name = 'mmd'+name_suffix)


        elif self.type in ['vamp', 'vamp_prior']:
            self.from_layer = ['act']
            self.from_output = ['act']
            return layers.VampNetwork(name = 'vamp', **self.loss_kwargs)
        

        elif self.type in ['iaf', 'iaf_encoder', 'iaf_conditional']:# 'iaf_density']:
            self.from_layer = ['addl']
            self.name = 'iaf'+name_suffix
            return Lambda(l.mean_one, arguments = {"keepdims": True}, name = self.name)

        elif self.type in ['maf', 'made_density', 'made_marginal', 'maf', 'maf_density']:
            self.from_layer = ['act']
            # specially handling density estimator
            self.type = 'maf_density'
            self.name = 'maf_density_reg_'+name_suffix
            return Lambda(l.dim_sum_one, arguments = {"keepdims": True}, name = self.name)
                
        else:
            # TRY IMPORT OF FUNCTION FROM LOSSES
            try:
                func = self._import_loss(self.type, 'keras.losses')
            except:
                print("Trying import ", self.type, " from l ")
                func = self._import_loss(self.type, 'losses')
            self.from_layer = ['stat']
            return Lambda(func, name = str(self.type+name_suffix))              


    def _import_loss(self, loss, module):
            try:
                mod = importlib.import_module(module)
                mod = getattr(mod, loss)
                print('loss function imported: ', mod, ' from ', module)
                return mod
            except:
                raise ValueError("Cannot import ", loss, " from ", module, '.  Please feed a valid loss function import or argument')


    def _get_loss_weight(self):
        if isinstance(self.weight, str):
            try:
                mod = importlib.import_module(LOSS_WEIGHTS)
                mod = getattr(mod, self.weight)
                loss_weight = mod(self.beta)
            except Exception as e:
                print(e) 
                raise ValueError("Cannot find weight loss function")
        elif isinstance(self.weight, float) or isinstance(self.weight, int) or ininstance(self.weight,  (tf.Variable, tf.Tensor)):
            loss_weight = self.weight
        else:
            raise ValueError("cannot interpret loss weight: ", self.weight) 
        return loss_weight




