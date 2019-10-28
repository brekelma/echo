import numpy as np
import tensorflow.keras.backend as K
import importlib
import tensorflow.keras.initializers
from tensorflow.keras.layers import Dense, Lambda, Reshape
import model_utils.layers as layers
#from model_utils.echo_noise import echo_sample
import tensorflow as tf
import copy
import model_utils.activations as activations

class Layer(object):
        def __init__(self, **kwargs):

                args = {'layer': -1,
                        'latent_dim': None,
                        'encoder': True,
                        'type': 'add', # can be Keras type or manual type from make_function_list
                        'k': 1, # e.g. iwae
                        'add_loss': True,
                        'activation': None, # custom activation (can be an import: module.___)
                        'data_input': False,
                        'density_estimator': None,
                        'noise_estimator': None,
                        'tc_estimator': None,
                        'layer_kwargs': {} # kw args for Keras or other layer
                }
                args.update(kwargs) # read kwargs into dictionary, set attributes beyond defaults
                for key in args.keys():
                        setattr(self, key, args[key])

                self.try_activations()

        def equals(self, other_args):
                return self.__eq__(other_args)


        def try_inits(self, default_init = 'glorot_uniform'):
                try:
                        init = self.layer_kwargs['init']
                except:
                        return default_init
                actstr = str(init)
                try:
                        mod = importlib.import_module(".".join(actstr.split(".")[:-1]))
                        act = getattr(mod, actstr.split(".")[-1])
                except:
                        try:
                                mod = importlib.import_module('tensorflow.keras.initializers')
                                act = getattr(mod, actstr)
                        except:
                                return default_init
                return act

        def try_activations(self, kw = False, key = None):
                if key is not None:
                        kw = True
                try:
                        act = self.layer_kwargs['activation'] if kw and key is None else self.activation
                        act = self.layer_kwargs[key] if key is not None else act
                        actstr = str(act)
                        mod = importlib.import_module('tensorflow.keras.activations')
                        act = getattr(mod, actstr)
                except Exception as e:
                        try:
                                mod = importlib.import_module(".".join(actstr.split(".")[:-1]))
                                act = getattr(mod, actstr.split(".")[-1])
                        except Exception as e:
                                pass
                                #import_str = str("model_utils."+(".".join(actstr.split(".")[:-1])))
                                #print(import_str)
                                #mod = importlib.import_module(import_str)
                                #act = getattr(mod, actstr.split(".")[-1])
                                #print("Couldn't import ", actstr.split(".")[-1], ": ", e)

                if not self.layer_kwargs.get('activation', True):
                        self.layer_kwargs[key if key is not None else 'activation'] = act
                elif self.layer_kwargs.get(key, False):
                        self.layer_kwargs[key] = act

        def make_function_list(self, index = 0):
                stats_list = []
                act_list = []
                addl_list = []
                
                for samp in range(self.k):
                        net = 'Enc' if self.encoder else 'Dec'
                        name_suffix = str(net)+'_'+str(index)+'_'+str(samp) if self.k > 1 else str(net)+'_'+str(index)
                        

                        if self.type in ['add', 'vae', 'additive']: 
                                if samp == 0:
                                        z_mean = Dense(self.latent_dim, activation='linear',
                                                                   name='z_mean'+name_suffix, kernel_initializer = self.try_inits())#, **self.layer_kwargs)
                                        z_logvar = Dense(self.latent_dim, activation='linear',
                                                           name='z_var'+name_suffix, 
                                                           kernel_initializer = self.try_inits())#, **self.layer_kwargs)            
                                        stats_list.append([z_mean, z_logvar])
                                
                                z_act = Lambda(layers.vae_sample, name = 'z_act_'+name_suffix) 
                                
                                act_list.append(z_act)
                                addl_list.append(z_act)
                                
                        elif self.type in ['mul', 'ido', 'multiplicative']: # information dropout
                                if samp == 0:   
                                        z_mean = Dense(self.latent_dim, activation='linear', name='z_mean'+name_suffix, **self.layer_kwargs)
                                        z_logvar = Dense(self.latent_dim, activation='linear', name='z_var'+name_suffix, **self.layer_kwargs)
                                        stats_list.append([z_mean, z_logvar])
                                z_act = Lambda(layers.ido_sample, name = 'z_act_'+name_suffix, arguments =self.layer_kwargs)
                                act_list.append(z_act)

                        # elif self.type in ['gated_linear', 'gated']:
                        #     w1 = Dense(self.latent_dim, activation = 'linear', name = 'w1'+name_suffix, **self.layer_kwargs)
                        #     w2 = Dense(self.latent_dim, activation = 'sigmoid', name = 'w2'+name_suffix, **self.layer_kwargs)

                        #     def my_multiply(inputs):
                        #         return tf.multiply(inputs[0], inputs[1])

                        #     out = Lambda(my_multiply, name = 'GatedLinear'+name_suffix)
                        #     stats_list.append([w1,w2])
                        #     act_list.append(out)
                                
                        elif self.type in ['echo']:
                                if self.layer_kwargs.get('per_sample', None) is None:
                                        self.layer_kwargs['per_sample'] = True  
                                else:
                                        raise NotImplementedError() # omitted

                                if samp == 0:   
                                        if self.layer_kwargs.get('fx_act', False):
                                                self.try_activations(key = 'fx_act')
                                                z_mean = Dense(self.latent_dim, activation=self.layer_kwargs['fx_act'], kernel_initializer = self.try_inits(), name='z_mean'+name_suffix)
                                        else:
                                                z_mean = Dense(self.latent_dim, activation=activations.tanh64,  kernel_initializer = self.try_inits(), name='z_mean'+name_suffix)
                                        

                                
                                        if self.layer_kwargs.get('sx_act', False):
                                                self.try_activations(key = 'sx_act')
                                                z_echo_scale = Dense(self.latent_dim, activation=self.layer_kwargs['sx_act'], name='z_sx'+name_suffix,  kernel_initializer = self.try_inits())#, bias_initializer = bias_init)# **self.layer_kwargs)
                                        else:
                                                z_echo_scale = Dense(self.latent_dim, activation=tf.math.log_sigmoid, name='z_sx'+name_suffix,  kernel_initializer = self.try_inits())#, bias_initializer = bias_init)# **self.layer_kwargs)
                                                        
                                        if self.layer_kwargs.get('init', None) is not None:
                                                del self.layer_kwargs['init']
                                        if self.layer_kwargs.get('bias', None) is not None:
                                                del self.layer_kwargs['bias']
        
                                        z_act = Lambda(layers.echo_sample, name = 'echo_act'+name_suffix, arguments = self.layer_kwargs)
                                                

                                stats_list.append([z_mean, z_echo_scale])
                                act_list.append(z_act)


                        elif self.type in ['iaf', 'inverse_flow']:
                                z_mean = Dense(self.latent_dim, activation='linear',
                                                          name='z_mean'+name_suffix)#**self.layer_kwargs)
                                z_logvar = Dense(self.latent_dim, activation='linear',
                                                          name='z_var'+name_suffix)# **self.layer_kwargs)            
                                stats_list.append([z_mean, z_logvar])

                                
                                self.try_activations(kw = True)
                                        

                                iaf = layers.IAF(name = 'z_act'+name_suffix,
                                        **self.layer_kwargs)

                                iaf_density = Lambda(iaf.get_density, name = 'iaf'+name_suffix)
                                iaf_base = Lambda(iaf.get_base, name = 'iaf_base'+name_suffix)
                                act_list.append(iaf)
                                
                                try:
                                        a = self.layer_kwargs['lstm']
                                        iaf_jac = Lambda(iaf.get_log_det_jac, name = 'iaf_jac'+name_suffix)
                                        if a:
                                                addl_list.append([iaf_density, iaf_base, iaf_jac])
                                except:
                                        addl_list.append([iaf_density, iaf_base])


                        elif self.type in ['maf']:
                                self.try_activations(kw = True)
                                
                                self.layer_kwargs["name"] = 'maf_density'+str(name_suffix)

                                if self.tc_estimator is not None:
                                        reshape = Lambda(layers.shuffle_batch)
                                else:
                                        reshape = Lambda(layers.list_id)

                                maf = layers.MAF(**self.layer_kwargs)   
                                
                                if self.layer_kwargs.get("add_base", True):
                                        base_density = Lambda(maf.get_base_output)
                                        addl_list.append(base_density)
                                act_list.append(maf)
                                
                                        
                                        
                        else:
                                # import layer module by string (can be specified either in activation or layer_kwargs)
                                try:
                                        spl = str(self.activation).split('.')
                                        if len(spl) > 1:
                                                path = '.'.join(spl[:-1])
                                                mod = importlib.import_module(path)
                                                mod = getattr(mod, spl[-1])
                                                self.layer_kwargs['activation'] = mod
                                        else:
                                                if not self.layer_kwargs.get('activation', True):
                                                        self.layer_kwargs['activation'] = self.activation
                                except Exception as e:
                                        pass

                                try:
                                        self.try_activations(kw = True)
                                except Exception as e:
                                        pass

                                try:
                                        spl = str(self.type).split('.')
                                        if len(spl) > 1:
                                                path = '.'.join(spl[:-1])
                                                mod = importlib.import_module(path)
                                                self.layer_kwargs['name'] = str(path+name_suffix)
                                                mod = getattr(mod, spl[-1])
                                                z_act = mod(self.latent_dim, **self.layer_kwargs)
                                        else:
                                                mod = importlib.import_module(str('tensorflow.keras.layers'))
                                                #mod = importlib.import_module(str('tensorflow.python.tensorflow.keras.layers'))
                                                self.layer_kwargs['name'] = str(self.type + name_suffix)
                                                
                                                if self.type == 'Dense':
                                                        z_act = Dense(self.latent_dim, **self.layer_kwargs)
                                                else:
                                                        z_act = getattr(mod, self.type)
                                                        try:
                                                                z_act = z_act(self.latent_dim, **self.layer_kwargs)
                                                        except:
                                                                z_act = z_act(**self.layer_kwargs)
                                except:
                                        raise AttributeError("Error Importing Activation ", self.type)
                                act_list.append(z_act)

                try:
                        return {'stat': stats_list, 'act': act_list, 'addl': addl_list, 'call_on_addl': call_on_addl}
                except:
                        return {'stat': stats_list, 'act': act_list, 'addl': addl_list}
