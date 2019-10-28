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
import pickle
from copy import copy
import h5py

import dataset
#import analysis

import model_utils.layers as layers
import model_utils.layer_args as layer_args
from model_utils.layers import IAF #Beta, MADE, MADE_network, IAF, Echo
import model_utils.losses as losses
from model_utils.loss_args import Loss
#import model_utils.sampling as sampling

#from model_utils.sonnet.moving_average import MovingAverage
#from model_utils.constraints import OptimizationConstraints

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten #Concatenate, 
from tensorflow.keras.layers import Activation, BatchNormalization, Lambda, Reshape 
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.callbacks import Callback, TensorBoard, TerminateOnNaN, EarlyStopping
from model_utils.callbacks import ZeroAnneal
from model_utils.callbacks import MyLearningRateScheduler as LearningRateScheduler
#import keras
import tensorflow.keras as keras
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
                                                'density_epochs': [],
                                                'eval_on_other': False,
                                                'callbacks': [],
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
                
                def encoder_model(self, conv_input = False):
                                enc_model = keras.models.Model(inputs = self.input if conv_input else self.input_tensor, outputs = self.encoder_layers[-1]['act'][0])
                                return enc_model
                def encoder_fx_model(self):
                                return keras.models.Model(inputs = self.input_tensor, outputs = self.encoder_layers[-1]['stat'][0][0])
                def encoder_sx_model(self):
                                return keras.models.Model(inputs = self.input_tensor, outputs = self.encoder_layers[-1]['stat'][0][-1])
                
                def decoder_model(self):
                        for i in range(len(self.model.layers)):
                                if 'act' in self.model.layers[i].name:
                                        lyr_ind = i
                        inp_shp = self.model.layers[lyr_ind].output_shape
                        #print(self.model.summary())
                        
                        z_inp = Input(inp_shp[1:], name = 'z_generator_inp')
                        
                        h = z_inp
                        for ii in range(lyr_ind+1, len(self.model.layers)):
                                print("MODEL LYR ", self.model.layers[ii](h))
                                if 'flatten' in self.model.layers[ii].name:
                                        break
                                h = self.model.layers[ii](h)
                                if 'pred' in self.model.layers[ii].name or 'Conv2DDec' in self.model.layers[ii].name:
                                        break
                                                           
                        self.decoder_model = keras.models.Model(inputs = z_inp, outputs = h)
                        print(self.decoder_model.summary())
                        return self.decoder_model

                def make_inputs(self):
                                if self.input_shape is None:
                                                self.input_shape = (self.dataset.dims[0], self.dataset.dims[1], 1) if 'Conv' in self.layers[0]['type'] else (self.dataset.dim,)

                                self.input_tensor = Input(shape = (self.dataset.dim,)) 
   
                                if self.input_shape is not None:
                                                self.input = Reshape(self.input_shape)(self.input_tensor)
                                else:
                                                self.input_shape = (self.dataset.dim,)
                                                self.input = self.input_tensor    
                                self.recon_true = self.input_tensor # Lambda(lambda y : y, name = 'x_true')(x)
   
                def build(self):
                                self.make_inputs()
                                self.encoder_layers = self._build_architecture([self.input], encoder = True)
                                self.encoder_mu = keras.models.Model(inputs = self.input_tensor, outputs = self.encoder_layers[-1]['stat'][0][0])
                                self.encoder_var = keras.models.Model(inputs = self.input_tensor, outputs = self.encoder_layers[-1]['stat'][0][-1])

                                self.decoder_layers = self._build_architecture(self.encoder_layers[-1]['act'], encoder = False)
                                self.model_outputs, self.model_losses, self.model_loss_weights = self._make_losses()

                                self.model = keras.models.Model(inputs = self.input_tensor, outputs = self.model_outputs)        

                def fit(self, x_train, y_train = None, x_val = None, y_val = None, verbose = None, validate = True, fit_gen = True, epochs = None):
                                if epochs is not None:
                                                self.epochs = epochs
                                
                                if verbose is not None:
                                                self.verbose = verbose

                                self.dc_outputs = []
                                #make_keras_pickleable()    

                                self.build() 
                                #self.make_inputs()
                                #self.encoder_layers = self._build_architecture([self.input], encoder = True)
                                #self.decoder_layers = self._build_architecture(self.encoder_layers[-1]['act'], encoder = False)
                                
                                
                                #self.encoder_model = keras.models.Model(inputs = self.input_tensor, outputs = self.encoder_layers[-1]['act'])

                                #self.encoder_mu = keras.models.Model(inputs = self.input_tensor, outputs = self.encoder_layers[-1]['stat'][0][0])
                                #self.encoder_var = keras.models.Model(inputs = self.input_tensor, outputs = self.encoder_layers[-1]['stat'][0][-1])
                                #self.encoder_act = keras.models.Model(inputs = self.input_tensor, outputs = self.encoder_layers[-1]['act'][0])

                                
                                #dec_input_tensor = Input(tensor = self.encoder_layers[-1]['act'][0])
                                #self.decoder_model = keras.models.Model(inputs = dec_input_tensor, outputs = self.decoder_layers[-1]['act'])

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

                                with self.sess.as_default():
                                                tf.global_variables_initializer().run()

                                callbacks = self._make_callbacks()
                                                #self.model.compile(optimizer = self.optimizer, loss = self.model_losses, loss_weights = self.model_loss_weights) # metrics?
                                tic = time.time()
                                
                                if self.epochs == 0:
                                                return

                                if self.lagrangian_fit:
                                                self._lagrangian_optimization(x_train, y_train, x_val, y_val)
                                else:
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

                                #np.random.seed(1)
                                np.random.seed(self.seed)
                                np.random.shuffle(self.dataset_clean.x_test)
                                self.test_eval(x_test = self.dataset_clean.x_test) 
                                
                                self.full_test_loss ={}
                                # get intermediate echo layers
                                if True: #try:
                                                self.z_fx = self.get_layer(self.dataset_clean.x_test, name = 'z_mean')
                                                self.z_sx = self.get_layer(self.dataset_clean.x_test, name = 'var')
                                                self.z_act = self.get_layer(self.dataset_clean.x_test, name = 'act')
                                                print()
                                                #for i in self.model.outputs:
                                                #        print("Model output ", i.name)
                                                #        self.full_test_loss[i.name] = self.get_layer(self.dataset_clean.x_test, name = i.name)
                                else: #except Exception as e:
                                                print("*"*50)
                                                print("Exception get layer: ", e)
                                                print("*"*50)
                                                #self.get_layer(self.dataset_clean.x_test, name = 'fx')

                                #self.decoder_model.save(self.filename+'_decoder.hdf5')
                                try:
                                                self.model.save_weights(self.filename+"_model_weights.h5")
                                except Exception as e:
                                                print(e)
                                                import IPython
                                                IPython.embed()

                                if self.eval_on_other:
                                                if 'binary' in self.dataset.name and 'mnist' in self.dataset.name:
                                                                others = [dataset.fMNIST(), dataset.Omniglot()]
                                                elif 'fmnist' in self.dataset.name:
                                                                others = [dataset.MNIST(binary = True), dataset.Omniglot()]
                                                else:
                                                                others = [dataset.MNIST(binary = True), dataset.fMNIST()]
                                                                #raise ValueError("What is the other dataset to evaluate?")

                                                for other in others:
                                                                self.test_eval(other.x_test, suffix = other.name)

                                                                for i in self.model.outputs:
                                                                                self.full_test_loss[i.name+'_'+other.name] = self.get_layer(other.x_test, name = i.name)

                                try:
                                                self.pickle_dump()
                                except:
                                                self.pickle_dump(small = True)

                                if self.encoder_dims[-1] == 2:
                                                analysis.plot_2d(enc_function = self.get_layer(name='act'), fn = self.filename, batch = self.batch)


                def pickle_dump(self, small = False):
                                stats = {}
                                for k in self.hist.keys():
                                                stats[k] = self.hist[k]
                
                                try:
                                                for k in self.test_results.keys():
                                                                stats[str('test_'+k)] = self.test_results[k]
                                except:
                                                pass
                                
                                try:
                                                if not small:
                                                                stats['z_fx']  = self.z_fx
                                                                stats['z_sx'] = self.z_sx
                                                                stats['z_act'] = self.z_act
                                                stats['test_loss_dict'] = self.full_test_loss
                                except Exception as e:
                                                print("*"*50)
                                                print('pickle dumping activations ', e)
                                                print('*'*50)

                                try:
                                                if isinstance(self.final_lagr, list):
                                                                for i in range(len(self.final_lagr)):
                                                                                stats['lagr_'+str(i)] = self.final_lagr[i]
                                                else:
                                                                stats['lagr'] = self.final_lagr
                                except Exception as e:
                                                print("No Lagrange multipliers")

                                with open(str(self.filename+".pickle"), "wb") as fle:
                                                pickle.dump(stats, fle)

                                #with open(str(self.filename+"_full_model.pickle"), "wb") as fle:
                                #        pickle.dump(self, fle)
                                                

                def test_eval(self, x_test = None, y_test = None, suffix = ''):
                                try:
                                                a = self.test_results
                                except:
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
                                                self.test_results[self.model.metrics_names[i]+'_'+suffix] = loss_list[i]
                                                print("Test loss ", self.model.metrics_names[i], " : ", loss_list[i])
                                return self.test_results

                def get_layer(self, x = None, name = 'act'):
                                for i in range(len(self.model.layers)):
                                                if name in self.model.layers[i].name:
                                                                lyr_ind = i
                                                                break
                                oputs = [self.model.layers[i].get_output_at(0)]
                                
                                new_m = keras.models.Model(inputs = self.model.inputs, outputs = oputs)
                                
                                if x is None:
                                        return new_m
                                else:
                                        try:
                                                        loss_list = new_m.predict(x, batch_size = self.batch)
                                        except:
                                                        try:
                                                                        rounded = x.shape[0] - x.shape[0] % self.batch
                                                                        loss_list = new_m.predict(x[:rounded,...], batch_size = self.batch)
                                                        except:
                                                                        loss_list = new_m.predict(x.reshape((x.shape[0], -1)), batch_size = self.batch)
                                        
                                        #try:
                                        #        print(loss_list.shape)
                                        #except:
                                        #        print([ll.shape for ll in loss_list])
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
                                                                                                
                                                                                                print(layer.type)
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
  
                                                                if 'discrim' in loss.type:
                                                                                
                                                                                self.discrim_vars = self.density_estimators['stat'][0]
                                                                                self.discrim_index = len(self.loss_functions)-1
                                                                #self.discrim_vars = self.loss_functions[-1].discrim_vars if loss.type in ['tc', 'tc_discrim'] else None
                                                                #self.discrim_index = len(self.loss_functions)-1

                                                                layer_inputs, output_inputs = loss.describe_inputs()
                                                                outputs = []
                                                                   
                                                                
                                                                for j in range(len(layer_inputs)): # 'stat' or 'act'
                                                                                # enc / dec already done
                                                                                layers = self.encoder_layers if enc else self.decoder_layers
                                                                                layers = layers if (not 'density' in loss.type and not 'tc_gan' in loss.type) else self.density_estimators
                                                                                lyr = loss.layer 
                                                                                
                                                                                
                                                                                if 'act' in layer_inputs[j] or 'addl' in layer_inputs[j]:
                                                                                                if 'list' in layer_inputs[j]: #isinstance(layers[lyr][layer_inputs[j]], list): 
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
                                                                                for j in range(len(outputs)):
                                                                                                if not isinstance(outputs[j], list):
                                                                                                                continue
                                                                                                
                                                                                                for k in range(len(outputs[j])):
                                                                                                                try:
                                                                                                                                outputs[j][k] = Flatten(name = 'flatten_'+outputs[j][k].name.split("/")[0]+'_'+str(i))(outputs[j][k]) if len(list(outputs[j][k].get_shape())) > 2 else outputs[j][k]
                                                                                                                except:
                                                                                                                                pass

                                                                
                                                                
                                                                
                                                                try:
                                                                                self.model_outputs.append(self.loss_functions[-1](outputs))
                                                                except Exception as e:
                                                                                temp_outputs = copy(outputs)
                                                                               
                                                                                if 'vamp' in loss.type:
                                                                                                temp_outputs[0] = outputs[0][-1] if len(outputs[0])<3 else outputs[0][-2:]
                                                                                                #temp_outputs[0] = outputs[0][-1]
                                                                                else:
                                                                                                temp_outputs[0] = outputs[0][0]
                                                                                
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
                                
                                for cb in self.callbacks:
                                                if isinstance(cb, str):
                                                                if 'early' in cb.lower() and 'stopping' in cb.lower():
                                                                                es = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto') # restore_best_weights = True
                                                                                my_callbacks.append(es)
                                print()
                                print("My Callbacks: ", my_callbacks)
                                print()
                                return my_callbacks


                def _decoder(self):
                                return self.decoder_model

                def _encoder(self):
                                return self.encoder_model

                def _parse_args(self):
                                # Note : can also pass dataset directly
                                if self.dataset == 'fmnist':
                                                d = dataset.fMNIST()
                                elif self.dataset == 'binary_fmnist':
                                                d = dataset.fMNIST(binary = True)
                                elif self.dataset == 'binary_mnist':
                                                d = dataset.MNIST(binary= True)
                                elif self.dataset == 'mnist':
                                                d = dataset.MNIST()
                                elif self.dataset in ['omniglot', 'omni']:
                                                d = dataset.Omniglot()
                                elif self.dataset == 'dsprites':
                                                d = dataset.DSprites()
                                elif self.dataset in ['dsprites_corr', 'ds_corr']:
                                                d = dataset.DSpritesCorrelated(remove_prob = 0.15, n_blocks = 4)
                                                #epoch_mult = int(1.0/(d.x_train.shape[0]/737280))                                                                                                                                                     
                                elif self.dataset == "cifar10" or self.dataset == 'cifar':
                                                d = dataset.Cifar10()


                                # selects per_label examples from each class for reduced training data
                                if self.per_label is not None:
                                                self.dataset.shrink_supervised(self.per_label)
                                self.dataset_clean = copy(self.dataset)

                                if self.optimizer_params.get("norm", False):
                                                opt_norm = self.optimizer_params["norm"]
                                                del self.optimizer_params["norm"]
                                else:
                                                opt_norm = None 
                                self.optimizer_name = self.optimizer
                                self.optimizer = getattr(optimizers, self.optimizer)(**self.optimizer_params)
                                #self.optimizer = getattr(tf.train, self.optimizer+'Optimizer')(**self.optimizer_params)
                                
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
                                                                                mod = importlib.import_module(str('model_utils.lr_sched'))
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
                                                                except Exception as e:
                                                                                print(e)
                                                                                self.config = json.load(open('configs/'+config))
                                                self.args.update(self.config)
                                                try:
                                                                with open(self.filename+'_config.json', 'w') as configjson:
                                                                                json.dump(self.config, configjson)
                                                except:
                                                                pass

                def _lagrangian_optimization(self,x_train, x_val = None, y_train = None, y_val = None, min_lagr = 0.0, max_lagr = 100.0, lr_lagr = None):
                                begin_vars = tf.trainable_variables()
                                
                                if self.constraints is not None:
                                                constraints = self._parse_constraints(min_lagr, max_lagr)
                                                total_loss = constraints() + tf.add_n([self.model_loss_weights[i]*tf.reduce_mean(losses.loss_val(self.model.outputs[i])) for i in range(len(self.model.outputs))]) 
                                else:
                                                total_loss = tf.add_n([self.model_loss_weights[i]*tf.reduce_mean(losses.loss_val(self.model.outputs[i])) for i in range(len(self.model.outputs))]) 
                                print("*"*10)
                                print("Constraints parsed, loss created")
                                print("*"*10)
                                learning_rate = tf.placeholder(tf.float32)
                                                
                                # TO DO : facilitate passing optimizer_params
                                if "Adam" in self.optimizer_name:
                                                trainer = tf.train.AdamOptimizer(learning_rate, **self.optimizer_params).minimize(total_loss)
                                elif "SGD" in self.optimizer_name:
                                                trainer = tf.train.GradientDescentOptimizer(learning_rate, **self.optimizer_params).minimize(total_loss)
                                else:
                                                trainer = tf.train.AdamOptimizer(learning_rate, **self.optimizer_params).minimize(total_loss)
                                
                                if self.discrim_vars is not None:
                                                discrim_trainer = tf.train.AdamOptimizer(learning_rate, **self.optimizer_params).minimize(losses.loss_val(self.model.outputs[self.discrim_ind]), var_list=self.discrim_vars)
                                else:
                                                discrim_trainer = tf.no_op()

                                
                                with self.sess.as_default():
                                                new_vars = list(set(tf.global_variables()) - set(begin_vars))
                                                tf.variables_initializer(new_vars).run()
                                                begin_vars = set(tf.global_variables())

                                n_samples = x_train.shape[0]
                                self.hist = defaultdict(list)
                                print("Begining TRAINING")
                                with self.sess.as_default():
                                                for epoch in range(self.epochs):
                                                                epoch_avg = defaultdict(list)
                                                                perm = np.random.permutation(n_samples) 
                
                                                                if self.anneal_functions is not None:
                                                                                for a in range(len(self.anneal_functions)):
                                                                                                try:
                                                                                                                ind = self.anneal_functions[a]['loss']
                                                                                                except:
                                                                                                                ind = self.anneal_functions[a]['ind']
                                                                                anneal =  tf.assign(self.model_loss_weights[ind], self.anneal_functions[a]['function'](epoch))
                                                                                self.sess.run(anneal)
                                                                
                                                                for offset in range(0, (int(n_samples / self.batch) * self.batch), self.batch):
                                                                                batch_data = x_train[perm[offset:(offset + self.batch)]]
                                                                                fd ={self.input_tensor: batch_data, 
                                                                                         learning_rate: self.lr if not self.lr_callback else self.lr(epoch)}

                                                                                to_run = [trainer, total_loss]+ self.model.outputs
                                                                                to_run = to_run + (constraints.lagrange_multipliers if self.constraints is not None else [])
                                                                                to_run = to_run + (discrim_trainer if self.discrim_vars is not None else [])
                                                                                tl_ind = 1

                                                                                result = self.sess.run(to_run, feed_dict = fd)

                                                                                epoch_avg['total_loss'].append(np.mean(result[tl_ind]))
                                                                                for loss_ind in range(len(self.model.outputs)):
                                                                                                batch_loss = result[loss_ind+tl_ind+1]
                                                                                                batch_loss = losses.loss_val(batch_loss)
                                                                                                epoch_avg[self.model.outputs[loss_ind].name].append(batch_loss)

                                                                # print and record epoch averages
                                                                print("Epoch ", str(epoch), ": ", end="") 
                                                                for loss_layer in epoch_avg.keys():#self.model.outputs:
                                                                                if epoch_avg[loss_layer] is not None and epoch_avg[loss_layer][-1] is not None:
                                                                                                epoch_loss = np.mean(epoch_avg[loss_layer]) if 'echo_noise_var' not in loss_layer else np.var(epoch_avg[loss_layer])
                                                                                                self.hist[loss_layer].append(epoch_loss)
                                                                                if self.verbose:
                                                                                                print(loss_layer.split("/")[0]+" : "+str(epoch_loss)+" \t ", end="")

                                                                if self.verbose:
                                                                                try:
                                                                                                print(" Lagrange Multipliers : ", self.sess.run(constraints.lagrange_multipliers, feed_dict = fd))
                                                                                except:
                                                                                                pass
                                                                print()
                                                self.final_lagr = self.sess.run(constraints.lagrange_multipliers, feed_dict = fd)
                                #constraints.lagrange_multipliers
                                                                
                
                def _parse_constraints(self, min_lagr = 0.0, max_lagr = 100.0):
                                #if self.constraints is not None:
                                                
                                # create constraints container (from sonnet)
                                constraints = OptimizationConstraints(valid_range=(min_lagr, max_lagr))
                                # each c in self.cosntraints is a dictionary
                                # c['loss'], c['value'], c['relation'] = 'geq', 'leq', c['init']
                                # c['min_lagr'/'max_lagr'], c['moving_avg']
                                for c in self.constraints:
                                                init = c.get('init', 1.0)
                                                
                                                # moving average of loss tensor?
                                                loss_tensor = tf.squeeze(losses.loss_val(self.model.outputs[c['loss']]))
                                                
                                                loss_ma = MovingAverage(decay=0.99, local = False)(loss_tensor)
                                               
                                                # constrained to be 
                                                constant = c['value']
                                                
                                                # logic to set constant according to loss_i
                                                if len(str(constant).split("_")) == 2:
                                                                val = self.model.outputs[int(constant.split("_")[-1])]
                                                                print("ENTERING LOSSES.val MOVING AVG")
                                                                constant_tensor = losses.loss_val(val)          
                                                                constant = MovingAverage(decay = .99, local = False)(constant_tensor)
                                                                
                                                if c['relation'] in ['geq', 'greater_than']:
                                                                constraints.add_geq(loss_ma, constant)
                                                elif c['relation'] in ['leq', 'less_than']:
                                                                constraints.add_leq(loss_ma, constant)
                                return constraints
                                
