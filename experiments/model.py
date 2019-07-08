import numpy as np
import time
t = time.time()
t= int(100*(t - int(t)))
np.random.seed(t)
import tensorflow as tf
tf.set_random_seed(t)


import importlib
import json
from collections import defaultdict
from keras import backend as K
import pickle
from copy import copy
import h5py

import dataset
import model_utils.layers as layers
import model_utils.layer_args as layer_args
from model_utils.layers import IAF #Beta, MADE, MADE_network, IAF, Echo
import model_utils.losses as losses
from model_utils.loss_args import Loss
import model_utils.sampling as sampling

import keras.backend as K
from keras.layers import Input, Dense, merge, Lambda, Flatten #Concatenate, 
from keras.layers import Activation, BatchNormalization, Lambda, Reshape 
import keras.optimizers
from keras.callbacks import Callback, TensorBoard, TerminateOnNaN
from model_utils.callbacks import ZeroAnneal
from model_utils.callbacks import MyLearningRateScheduler as LearningRateScheduler
#from keras_normalized_optimizers.optimizer import NormalizedOptimizer



class Model(object):
        def __init__(self, dataset, *args, **kwargs):
                hyper_params.update(kwargs)
                for key in hyper_params.keys():
                        self.__setattr__(key, hyper_params[key])

class NoiseModel(Model):
        def __init__(self, dataset, config = None, filename = 'latest_unnamed_model', seed = 12, verbose = True, args_dict = {}):
                self.filename = filename
                self.verbose = verbose
                self.input_shape = None
                self.dataset = dataset
                self.seed = seed
                

                self.args = {
                        'epochs': 100,
                        'batch': 100,
                        'input_shape': None,
                        'optimizer': 'Adam',
                        'optimizer_params': {},
                        'initializer': 'glorot_uniform',
                        'lr': 0.001,
                        'lr_echo': 0.01,
                        'lr_lagr': 0.01,
                        'lr_density': 0.001,
                        'dense_act': {'encoder': 'softplus', 'decoder': 'softplus'},
                        'output_activation': 'sigmoid',
                        'encoder_dims': None, #[200, 200, 50],
                        'decoder_dims': None,
                        'layers': None,
                        'recon': None,
                        'losses': None,
                        'constraints': None,
                        'lagrangian_fit': False,
                        'mismatch': None,
                        'beta': 1.0,
                        'per_label': None,
                        'anneal_schedule': None,
                        'anneal_functions': None,
                        'density_epochs': []
                }

                cp = tf.ConfigProto()
                cp.gpu_options.allow_growth = True
                self.sess = tf.Session(config = cp)
                K.set_session(self.sess)


                self._load_config(config) # updates arguments

                self.args.update(args_dict) 
                for key in self.args.keys():
                        setattr(self, key, self.args[key])

                self._parse_args()
                self._parse_layers_and_losses()
                #self._enc_ind_args = []

                # initialize dictionary (with keys = layer index) of dict of called layers (keys = 'stat', 'act')
                self.encoder_layers = [] #defaultdict(dict)
                self.decoder_layers = [] #defaultdict(dict)
                self.density_estimators = [] #defaultdict(list)
                self.density_callback = False
        
        def encoder_model(self):
                return keras.models.Model(inputs = self.input_tensor, outputs = self.encoder_layers[-1]['act'])
        def encoder_fx_model(self):
                return keras.models.Model(inputs = self.input_tensor, outputs = self.encoder_layers[-1]['stat'][0])
        def encoder_sx_model(self):
                return keras.models.Model(inputs = self.input_tensor, outputs = self.encoder_layers[-1]['stat'][0])

        def fit(self, x_train, y_train = None, x_val = None, y_val = None, verbose = None, validate = True, fit_gen = True):
                if verbose is not None:
                        self.verbose = verbose

                self.dc_outputs = []
                #make_keras_pickleable()    
                if self.input_shape is None:
                        self.input_shape = (self.dataset.dims[0], self.dataset.dims[1], 1) if 'Conv' in self.layers[0]['type'] else (self.dataset.dim,)

                self.input_tensor = Input(shape = (self.dataset.dim,)) 
   
                if self.input_shape is not None:
                        self.input = Reshape(self.input_shape)(self.input_tensor)
                else:
                        self.input_shape = (self.dataset.dim,)
                        self.input = self.input_tensor    
                self.recon_true = self.input_tensor # Lambda(lambda y : y, name = 'x_true')(x)
   
                self.encoder_layers = self._build_architecture([self.input], encoder = True)
                
                
                
                
                
                
                #self.encoder_model = keras.models.Model(inputs = self.input_tensor, outputs = self.encoder_layers[-1]['act'])

                self.encoder_mu = keras.models.Model(inputs = self.input_tensor, outputs = self.encoder_layers[-1]['stat'][0][0])
                self.encoder_var = keras.models.Model(inputs = self.input_tensor, outputs = self.encoder_layers[-1]['stat'][0][-1])
                
                self.decoder_layers = self._build_architecture(self.encoder_layers[-1]['act'], encoder = False)
                #dec_input_tensor = Input(tensor = self.encoder_layers[-1]['act'][0])
                #self.decoder_model = keras.models.Model(inputs = dec_input_tensor, outputs = self.decoder_layers[-1]['act'])

                self.model_outputs, self.model_losses, self.model_loss_weights = self._make_losses()

                self.model = keras.models.Model(inputs = self.input_tensor, outputs = self.model_outputs)

                self.model_loss_weights = [tf.Variable(lw, dtype = tf.float32, trainable = False) for lw in self.model_loss_weights]
                
                #if not self.lagrangian_fit: # OMITTED
                self.model.compile(optimizer = self.optimizer, loss = self.model_losses, loss_weights = self.model_loss_weights) # metrics?
                

                print(self.model.summary())
                print()
                print("Losses:")
                for i in range(len(self.model_loss_weights)):
                        print(self.model_outputs[i].name.split('/')[0], ": ", self.model_loss_weights[i])
                print()
                print("Layers:")
                for i in self.model.layers[1:]:
                        try:
                                print(i.name, i.activation)
                        except:
                                pass
                #self.encoder_model.compile(optimizer = self.optimizer, loss = self.model_losses[0])

                self.sess = tf.Session()
                with self.sess.as_default():
                        tf.global_variables_initializer().run()

                callbacks = self._make_callbacks()
                        #self.model.compile(optimizer = self.optimizer, loss = self.model_losses, loss_weights = self.model_loss_weights) # metrics?
                tic = time.time()
                if not fit_gen: #int(x_train.shape[0]/self.batch) == x_train.shape[0]/self.batch:
                        hist = self.model.fit(x_train, ([x_train] if y_train is None else [y_train])*len(self.model_outputs), 
                                                          epochs = self.epochs, batch_size = self.batch, callbacks = callbacks, verbose = self.verbose,
                                                                  validation_data = (self.dataset.x_test, [self.dataset.x_test]*len(self.model_outputs)) if validate else None)
                else:
                        # fit generator
                        hist = self.model.fit_generator(self.dataset.generator(batch = self.batch, target_len = len(self.model_outputs), mode = 'train', unsupervised = True),
                                                                                        steps_per_epoch = int(self.dataset.x_train.shape[0]/self.batch), epochs = self.epochs, callbacks = callbacks,
                                                                                        validation_data = self.dataset.generator(batch = self.batch, target_len = len(self.model_outputs), mode = 'test', unsupervised = True) 
                                                                                        if validate else None, validation_steps = int(self.dataset.x_test.shape[0]/self.batch) if validate else None)
        
 
                self.hist= hist.history

                self.fit_time = time.time() - tic

                np.random.seed(42)
                np.random.shuffle(self.dataset_clean.x_test)
                self.test_eval(x_test = self.dataset_clean.x_test) 
                

                # get intermediate echo layers
                try:
                        self.z_fx = self.get_layer(self.dataset_clean.x_test, name = 'z_mean')
                        self.z_sx = self.get_layer(self.dataset_clean.x_test, name = 'var')
                        self.z_act = self.get_layer(self.dataset_clean.x_test, name = 'act')
                except Exception as e:
                        print("*"*50)
                        print("Exception get layer: ", e)
                        print("*"*50)
                        #self.get_layer(self.dataset_clean.x_test, name = 'fx')

                #self.decoder_model.save(self.filename+'_decoder.hdf5')
                self.model.save_weights(self.filename+"_model_weights.hdf5")
                self.pickle_dump()
   

        def pickle_dump(self):
                stats = {}
                for k in self.hist.keys():
                        stats[k] = self.hist[k]
        
                try:
                        for k in self.test_results.keys():
                                stats[str('test_'+k)] = self.test_results[k]
                except:
                        pass
                
                try:
                        stats['z_fx']  = self.z_fx
                        stats['z_sx'] = self.z_sx
                        stats['z_act'] = self.z_act
                except Exception as e:
                        print("*"*50)
                        print('pickle dumping activations ', e)
                        print('*'*50)

                with open(str(self.filename+".pickle"), "wb") as fle:
                        pickle.dump(stats, fle)
                        

        def test_eval(self, x_test = None, y_test = None):
                self.test_results = {}
                if x_test is None:
                        x_test = self.dataset.x_test
                try:
                        find_recon = [i for i in range(len(self.model.metrics_names)) if 'recon' in self.model.metrics_names[i]]
                except:
                        find_recon = [-1]

                try:
                        loss_list = self.model.evaluate(x_test, [x_test]*len(self.model.outputs), batch_size = self.batch)
                except:
                        rounded = x_test.shape[0] - x_test.shape[0] % self.batch
                        loss_list = self.model.evaluate(x_test[:rounded,...], [x_test[:rounded,...]]*len(self.model.outputs), batch_size = self.batch)

                for i in range(len(loss_list)):
                        self.test_results[self.model.metrics_names[i]] = loss_list[i]
                        print("Test loss ", self.model.metrics_names[i], " : ", loss_list[i])


        def get_layer(self, x, name = 'act'):
                for i in range(len(self.model.layers)):
                        if name in self.model.layers[i].name:
                                lyr_ind = i
                                break
                print("INPUTS ", self.model.inputs)
                oputs = [self.model.layers[i].get_output_at(0)]
                print("proposed outputs ", oputs)
                new_m = keras.models.Model(inputs = self.model.inputs, outputs = oputs)
                
                try:
                        loss_list = new_m.predict(x, batch_size = self.batch)
                except:
                        rounded = x.shape[0] - x.shape[0] % self.batch
                        loss_list = new_m.predict(x[:rounded,...], batch_size = self.batch)
                print('get layer ', name, ' : ', type(loss_list))
                
                try:
                        print(loss_list.shape)
                except:
                        print([ll.shape for ll in loss_list])
                return loss_list
                


        def _build_architecture(self, input_list = None, encoder = True):
                if encoder:
                        layers_list = self.encoder_layers
                        dims = self.encoder_dims
                        ind_latent = self._enc_latent_ind
                        offset = 0

                else:
                        layers_list = self.decoder_layers
                        dims = self.decoder_dims
                        ind_latent = self._dec_latent_ind
                        offset = len(self.encoder_layers)
                
                for layer_ind in range(len(dims)):
                        _density = None
                        _noise = None
                        _tc = None
                        
                        if layer_ind == 0:
                                if input_list is not None:
                                        current_call = input_list
                                else:
                                        raise ValueError("Feed input tensor to _build_architecture")
                        else:
                                try:
                                        current_call = layers_list[layer_ind-1]['act']
                                except:
                                        current_call = self.decoder_layers[layer_ind-1]['act']
                                        
                        if layer_ind in ind_latent:
                                # retrieve layer index for given encoding layer
                                arg_ind = ind_latent.index(layer_ind) + offset
                                if self.layers[arg_ind].get('latent_dim', None) is None:
                                        self.layers[arg_ind]['latent_dim'] = dims[layer_ind]
                        
                                if "echo" in self.layers[arg_ind]['type'] or self.layers[arg_ind]['type'] in ['bir', 'constant_additive']:
                                        self.layers[arg_ind]['layer_kwargs']['batch'] = self.batch
                                                        
                                try:
                                        # set density estimator if applicable
                                        self.layers[arg_ind]['density_estimator']['latent_dim'] = self.layers[arg_ind]['latent_dim']
                                        _density = layer_args.Layer(**self.layers[arg_ind]['density_estimator']) 
                                except:
                                        pass
                                
                                try:
                                        # set tc estimator if applicable
                                        self.layers[arg_ind]['tc_estimator']['latent_dim'] = self.layers[arg_ind]['latent_dim']
                                        _tc = layer_args.Layer(**self.layers[arg_ind]['tc_estimator']) 
                                except:
                                        pass
                                
                                try:
                                        # set noise estimator if applicable
                                        self.layers[arg_ind]['noise_estimator']['latent_dim'] = self.layers[arg_ind]['latent_dim']
                                        _noise = layer_args.Layer(**self.layers[arg_ind]['noise_estimator']) 
                                        _noise.noise_estimator = True
                                except:
                                        pass

                                # Create Layer
                                layer = layer_args.Layer(** self.layers[arg_ind])
                        else:
                                # default Dense layer
                                if encoder:
                                        act = self.activation['encoder']
                                else:
                                        act = self.activation['decoder'] if not layer_ind == len(dims)-1 else self.output_activation

                                layer = layer_args.Layer(** {'type': 'Dense',
                                                                                'latent_dim': dims[layer_ind],
                                                                                'encoder': encoder,
                                                                                'layer_kwargs': {'activation': act,
                                                                                        'kernel_initializer': self.initializer,
                                                                                        'bias_initializer': self.initializer}})

                        for mode in [0,1]:
                                loop_noise = False
                                # mode 1 for density estimation
                                if mode == 1:
                                        if not (_density is not None or _noise is not None or _tc is not None):
                                                continue
                                        if _density is not None:
                                                functions = _density.make_function_list(index = layer_ind)
                                                layers_list = self.density_estimators
                                        if _tc is not None:
                                                tc_functions = _tc.make_function_list(index = layer_ind)
                                                layers_list = self.density_estimators
                                        if _noise is not None:   
                                                loop_noise = True
                                                noise_functions = _noise.make_function_list(index = layer_ind)
                                                layers_list = self.density_estimators
                                else:   
                                        layers_list = self.encoder_layers if encoder else self.decoder_layers
                                        functions = layer.make_function_list(index = layer_ind)


                                fns = [functions] if not loop_noise else [functions, noise_functions]

                                loops = ['default/density', 'noise', 'tc'] if loop_noise else ['default/density']
                                for loop in loops:
                                        enc_dec = self.encoder_layers if encoder else self.decoder_layers
                                        if loop == 'noise':
                                                functions = noise_functions
                                                density_ind = _noise
                                                layers_list.append(defaultdict(list))
                                                current_call = enc_dec[layer_ind]['addl'][-1]
                                                current = current_call
                                        elif _tc is not None and loop == 'tc':
                                                functions = tc_functions       
                                                current_call = enc_dec[layer_ind]['act']
                                                layers_list.append(defaultdict(list))
                                        else:
                                                if mode == 1 and _density is not None:
                                                                current_call = enc_dec[layer_ind]['act']
                                                functions = functions
                                                layers_list.append(defaultdict(list))
                                                
                                        stat_k = functions['stat']
                                        act_k = functions['act']
                                        addl_k = functions['addl']
                                        try:
                                                call_on_addl = functions['call_on_addl']
                                        except:
                                                call_on_addl = None

                                        if stat_k: # call stats layers (e.g. [z_mean, z_logvar] for VAE )
                                                current = current_call[0]
                                                intermediate_stats = []
                                                for k in range(len(stat_k)):
                                                        stat_lyr = stat_k[k]
                                                        for z in stat_lyr:
                                                                intermediate_stats.append(z(current))
                                                        try:
                                                                layers_list[layer_ind]['stat'].append(intermediate_stats)
                                                                if loop != 'noise':
                                                                                current_call = layers_list[layer_ind]['stat']
                                                        except: 
                                                                layers_list[-1]['stat'].append(intermediate_stats)
                                                                if loop != 'noise':
                                                                                current_call = layers_list[-1]['stat']
                                                                
                                        try:
                                                if len(act_k) < len(current_call):
                                                                act_k = [a for idx in range(int(len(current_call)/len(act_k))) for a in act_k]
                                        except:
                                                pass

                                        for k in range(len(act_k)):
                                                act_lyr = act_k[k]
                                        
                                                if isinstance(current_call, list):
                                                        current = current_call[k] if len(current_call) > 1 else current_call[0]
                                                
                                                if layer.data_input:
                                                        print("Layer ", layer.type, ' data input TRUE:  current call ', current)
                                                        current = [self.input, current]
                                                        print('new current ', current)
 
                                                act = act_lyr(current) # if not echo_flag else act_lyr(current)[0]
                                                
                                                try:
                                                        layers_list[layer_ind]['act'].append(act)
                                                        if mode == 1 and loop == 'noise': 
                                                                        current_call = layers_list[layer_ind]['addl']
                                                        elif mode == 1:
                                                                        current_call = layers_list[layer_ind]['act']

                                                except Exception as e:
                                                        print("Activation call exception for layer ", layer_ind," of len ", len(layers_list), "OR ", len(self.encoder_dims)," type ", self.layers[arg_ind]["type"])
                                                        layers_list[-1]['act'].append(act)
                                                        
                                                        if isinstance(act, IAF):
                                                                current_call = layers_list[-1]['act']
                                
                                        if not mode == 1:
                                                for k in range(len(addl_k)):
                                                        addl_lyr = addl_k[k]

                                                        if isinstance(current_call, list):
                                                                current = current_call[k] if len(current_call) > 1 else current_call[0]
                                                        else:
                                                                current = [current]
                                                        
                                                        if isinstance(addl_lyr, list):
                                                                for addl in addl_lyr:
                                                                        a = addl(current)
                                                          
                                                                        try:
                                                                                layers_list[layer_ind]['addl'].append(a)
                                                                        except Exception as e:
                                                                                print("Addl exception ", e)
                                                                                layers_list[-1]['addl'].append(a)
                                                                print()
                                                                print("APPENDING to addl")
                                                                print(layers_list[layer_ind]['addl'])
                                                                print()
                                                        else:
                                                                a = addl_lyr(current)
                                                                try:
                                                                        layers_list[layer_ind]['addl'].append(a)
                                                                except:
                                                                        layers_list[-1]['addl'].append(a)


                                        if call_on_addl is not None and call_on_addl:
                                                for k in [-1]:#range(len(call_on_addl)):
                                                        if isinstance(current_call, list):
                                                                current = current_call[k] if len(current_call) > 1 else current_call[0]
                                                        
                                                        if isinstance(call_on_addl, list):
                                                                for addl in call_on_addl:
                                                                        a = addl(current)
                                                                        try:
                                                                                layers_list[layer_ind]['act'].append(a)
                                                                        except:
                                                                                layers_list[-1]['act'].append(a)

                                                        else:
                                                                a = call_on_addl(current)
                                                                try:
                                                                                layers_list[layer_ind]['act'].append(a)
                                                                except:
                                                                                layers_list[-1]['act'].append(a)

                        if encoder:
                                layers_list = self.encoder_layers
                        else:
                                layers_list = self.decoder_layers
                        
                return layers_list



        def _make_losses(self, metrics = False):
                loss_list = self.losses
                self.model_outputs = []
                self.model_losses = []
                self.model_loss_weights = []


                if loss_list is not None:     
                        for i in range(len(loss_list)):
                                callback = False     
                                loss = Loss(**loss_list[i]) if isinstance(loss_list[i], dict) else loss_list[i]
           
                                enc = loss.encoder

                                if loss.type in ['mi_marginal', 'mi_joint']:
                                        loss.batch = self.batch
                                elif loss.type in ['vamp']:
                                        # initialize VampPrior with the data mean
                                        loss.loss_kwargs['init'] = np.mean(self.dataset.x_train.reshape(-1, self.dataset.dim), axis = 0)
                                        loss.loss_kwargs['encoder_mu'] = self.encoder_mu
                                        loss.loss_kwargs['encoder_var'] = self.encoder_var
                                try:
                                        self.loss_functions.append(loss.make_function())
                                except:
                                        self.loss_functions = []
                                        self.loss_functions.append(loss.make_function())
  
                                layer_inputs, output_inputs = loss.describe_inputs()
                                outputs = []
                                   
                                for j in range(len(layer_inputs)): # 'stat' or 'act'
                                        # enc / dec already done
                                        layers = self.encoder_layers if enc else self.decoder_layers
                                        layers = layers if not 'density' in loss.type and 'tc_gan' not in loss.type else self.density_estimators
                                        
                                        lyr = loss.layer 
                                        

                                        
                                        if 'act' in layer_inputs[j] or 'addl' in layer_inputs[j]:
                                                if 'list' in layer_inputs[j]: #isinstance(layers[lyr][layer_inputs[j]], list): 
                                                        print()
                                                        print("Appending ", layer_inputs[j])
                                                        print(layers[lyr])
                                                        print()
                                                        outputs.append(layers[lyr][layer_inputs[j].split("_")[0]])
                                                else:
                                                        outputs.extend(layers[lyr][layer_inputs[j]])
                                                #outputs.extend(layers[lyr][layer_inputs[j]])
                                        elif 'stat' in layer_inputs[j]:
                                                try:
                                                        # stat is a list of lists 
                                                        outputs.extend(layers[lyr][layer_inputs[j]][0])
                                                except:
                                                        outputs.extend(layers[-1][layer_inputs[j]])

                                for j in range(len(output_inputs)):

                                        layers = self.decoder_layers if not loss.output_density else self.density_estimators
                                        layers = layers if not 'density' in loss.type and not 'tc_gan' in loss.type else self.density_estimators
                                        lyr = loss.output
                                        
                                        if 'act' in output_inputs[j]: 
                                                # all output activations get either recon_true or output activations of some encoding layer
                                                if (lyr == -1): #[-1, len(self.decoder_dims)-1]):
                                                        recon_true = self.recon_true 
                                                else:
                                                        if len(self.encoder_layers[lyr]['act']) == 1:
                                                                recon_true = layers[lyr]['act'][0]
                                                        else:
                                                                raise NotImplementedError("Cannot handle > 1 activation for intermediate layer reconstruction")
                                                
                                                mylyr = copy(layers[lyr][output_inputs[j]])
                                                mylyr.insert(0, recon_true)
                                                outputs.extend(mylyr)
                                                
                                        elif 'stat' in output_inputs[j]:
                                                outputs.extend(layers[lyr][output_inputs[j]][0])
                                        # not handling 'addl' tensors

                                try:
                                         for j in range(len(outputs)):
                                                 outputs[j] = Flatten(name = 'flatten_'+outputs[j].name.split("/")[0]+'_'+str(i))(outputs[j]) if len(list(outputs[j].get_shape())) > 2 else outputs[j]
                                except Exception as e:
                                        print()
                                        print("EXCEPTION ", e)
                                        for j in range(len(outputs)):
                                                if not isinstance(outputs[j], list):
                                                        continue
                                                
                                                for k in range(len(outputs[j])):
                                                        try:
                                                                outputs[j][k] = Flatten(name = 'flatten_'+outputs[j][k].name.split("/")[0]+'_'+str(i))(outputs[j][k]) if len(list(outputs[j][k].get_shape())) > 2 else outputs[j][k]
                                                        except:
                                                                pass

                                
                                
                                
                                try:
                                        print()
                                        print("Model loss call ", loss.type)
                                        print(outputs)
                                        print()
                                        self.model_outputs.append(self.loss_functions[-1](outputs))
                                except:
                                        print()
                                        print()
                                        print("TEMP OUTPUTS before")
                                        temp_outputs = copy(outputs)
                                        print(temp_outputs)
                                        if 'vamp' in loss.type:
                                                temp_outputs[0] = outputs[0][-1]
                                        else:
                                                temp_outputs[0] = outputs[0][0]
                                        print("Temp outputs for : ", loss.type, temp_outputs)
                                        self.model_outputs.append(self.loss_functions[-1](temp_outputs))

                                self.model_losses.append(losses.dim_sum)
                                self.model_loss_weights.append(loss.get_loss_weight())
                                try:
                                        self.loss_inputs.append(outputs)
                                except:
                                        self.loss_inputs = []
                                        self.loss_inputs.append(outputs)

                        def reg_losses(tensors):
                                return [K.sum(t, keepdims=True) for t in tensors if 'reg' in t.name]


                return self.model_outputs, self.model_losses, self.model_loss_weights


        def _make_callbacks(self):
                my_callbacks = []
                if self.lr_callback:
                        my_callbacks.append(LearningRateScheduler(self.lr))
                my_callbacks.append(TerminateOnNaN())
                
                return my_callbacks


        def _decoder(self):
                return self.decoder_model

        def _encoder(self):
                return self.encoder_model

        def _parse_args(self):
                # Note : can also pass dataset directly
                if isinstance(self.dataset, str):
                        if self.dataset == 'mnist':
                                self.dataset = dataset.MNIST(binary = False)
                        elif self.dataset == 'binary_mnist':
                                self.dataset = dataset.MNIST(binary = True)
                        elif self.dataset == 'omniglot':
                                self.dataset = dataset.Omniglot()
                        elif self.dataset == 'celeb_a':
                                pass
                        elif self.dataset == 'dsprites':
                                self.dataset = dataset.DSprites()

                # selects per_label examples from each class for reduced training data
                if self.per_label is not None:
                        self.dataset.shrink_supervised(self.per_label)
                self.dataset_clean = copy(self.dataset)

                if self.optimizer_params.get("norm", False):
                        opt_norm = self.optimizer_params["norm"]
                        del self.optimizer_params["norm"]
                else:
                        opt_norm = None 
                self.optimizer = getattr(keras.optimizers, self.optimizer)(**self.optimizer_params)
                
                #if opt_norm is not None:
                #    self.optimizer = NormalizedOptimizer(self.optimizer, normalization = opt_norm)


                self.lr_callback = False
                if isinstance(self.lr, str):
                        try:
                                mod = importlib.import_module(str('lr_sched'))
                                # LR Callback will be True /// self.lr = function of epochs -> lr
                                self.lr_callback = isinstance(self.lr, str)
                                self.lr = getattr(mod, self.lr)
                        except:
                                try:
                                        mod = importlib.import_module(str('custom_functions.lr_sched'))
                                        # LR Callback will be True /// self.lr = function of epochs -> lr
                                        self.lr_callback = isinstance(self.lr, str)
                                        self.lr = getattr(mod, self.lr)
                                except:
                                        #self.lr = dflt.get('lr', .001)
                                        #print()
                                        warnings.warn("Cannot find LR Schedule function.  Proceeding with default, constant learning rate of 0.001.")    
                                        #print()

                # Architecture Args
                if self.encoder_dims is None:
                        try:
                                self.encoder_dims = self.latent_dims
                        except Exception as e:
                                print(e)
                                raise ValueError

                if self.decoder_dims is None:
                        self.decoder_dims = list(reversed(self.encoder_dims[:-1]))
                        self.decoder_dims.append(self.dataset.dim)
                else:
                        pass

        def _parse_layers_and_losses(self):
                # list of indices of regularized or custom layers
                self._enc_latent_ind = []
                self._dec_latent_ind = []
                

                # loop through to record which layers have losses attached
                if self.losses is not None and isinstance(self.losses, list):
                        pass
                        
                # loop through to record which layers have special layer arguments 
                # (noise regularization or non-dense layer type)
                for i in range(len(self.layers)):
                        layerargs = self.layers[i]
                        if layerargs.get('encoder', False):
                                # for each entry in layer args list, record corresponding index in latent_dims
                                self._enc_latent_ind.append(len(self.encoder_dims)-1 
                                                                                                if layerargs.get('layer', -1) == -1
                                                                                                else layerargs['layer'])
 
                        else:
                                self._dec_latent_ind.append(len(self.decoder_dims)-1
                                                                                                if layerargs.get('layer', -1) == -1
                                                                                                else layerargs['layer']) 



                if self.recon is not None:
                        #self._dec_loss_ind.append(len(self.decoder_dims)-1)
                        recon_loss = Loss(**{'type': self.recon,
                                                                'layer': -1,
                                                                'encoder': False,
                                                                'weight': 1
                                                        })
                        self.losses.append(recon_loss)

        def _load_config(self, config):
                if config is not None:
                        if isinstance(config, dict):
                                self.config = config
                        else:
                                try:
                                        self.config = json.load(open(config))
                                except:
                                        self.config = json.load(open('configs/'+config))
                        self.args.update(self.config)
                        try:
                                with open(self.filename+'_config.json', 'w') as configjson:
                                        json.dump(self.config, configjson)
                        except:
                                pass
