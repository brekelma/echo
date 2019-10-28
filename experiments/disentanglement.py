import importlib
from collections import defaultdict
import numpy as np
import model
import dataset
import argparse
import json
import pickle
#import disentanglement.evaluate
import os
import copy
import model_utils.losses as losses 
import tensorflow.keras.backend as K
import tensorflow.keras as keras
#from evaluation import named_data
#from evaluation.metrics import beta_vae  # pylint: disable=unused-import
#from evaluation.metrics import dci  # pylint: disable=unused-import
#from evaluation.metrics import downstream_task  # pylint: disable=unused-import
from evaluation.metrics import factor_vae  # pylint: disable=unused-import
#from evaluation.metrics import irs  # pylint: disable=unused-import
from evaluation.metrics import mig  # pylint: disable=unused-import
#from evaluation.metrics import modularity_explicitness  # pylint: disable=unused-import
#from evaluation.metrics import reduced_downstream_task  # pylint: disable=unused-import
#from evaluation.metrics import sap_score  # pylint: disable=unused-import
from evaluation.metrics import unsupervised_metrics  # pylint: disable=unused-import
from evaluation import results
from evaluation import dsprites
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='test_config.json')
parser.add_argument('--noise', type=str)
parser.add_argument('--folder', type=str)
parser.add_argument('--filename', type=str)
parser.add_argument('--count', default=0)
parser.add_argument('--count_offset', default=0)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--validate', type=int, default = 0)
parser.add_argument('--verbose', type=int, default = 1)
parser.add_argument('--fit_gen', type=bool, default = 1)
parser.add_argument('--seed', type=int, default = 12)
parser.add_argument('--batch', type=int, default = 64)
parser.add_argument('--evals', nargs='+', type=str)
parser.add_argument('--epochs', type=int)
parser.add_argument('--per_label')
parser.add_argument('--remove_prob', type=float, default=0.15)
parser.add_argument('--blocks', type=int, default=4)
parser.add_argument('--shrink')
parser.add_argument('--gpu')
parser.add_argument('--tc_weight', default=0)
parser.add_argument('--stop_for_disent', default=1)
parser.add_argument('--dataset', type=str, default = 'dsprites')
args, _ = parser.parse_known_args()

#if args.gpu is not None:
#        gpu_options = tf.GPUOptions(visible_device_list=str(args.gpu))
#        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

if ".json" in args.config:
        config = args.config if not 'ds' in args.dataset and not 'dsprites' in args.dataset else 'dsprites/'+args.config
else:
        config = json.loads(args.config.replace("'", '"'))

args.count = int(args.count)+int(args.count_offset)
args.beta = float(args.beta)
args.tc_weight = float(args.tc_weight)


print("ARGS DATASET ", args.dataset)
if args.dataset == 'fmnist':
        d = dataset.fMNIST()
elif args.dataset == 'binary_fmnist':
        d = dataset.fMNIST(binary = True)
elif args.dataset == 'binary_mnist':
        d = dataset.MNIST(binary= True)
elif args.dataset == 'mnist':
        d = dataset.MNIST()
elif args.dataset in ['omniglot', 'omni']:
        d = dataset.Omniglot()
elif args.dataset == 'dsprites':
        d = dataset.DSprites()
elif args.dataset in ['dsprites_corr', 'ds_corr']:
        d = dataset.DSpritesCorrelated(remove_prob = args.remove_prob, n_blocks = args.blocks)
        # Train same number of gradient steps for downsampled dataset
        epoch_mult = int(1.0/(d.x_train.shape[0]/737280))
elif args.dataset == "cifar10" or args.dataset == 'cifar':
        d = dataset.Cifar10()

if args.per_label is not None:
        d.shrink_supervised(args.per_label)
if args.shrink is not None:
        d.x_train = d.x_train[:shrink]
        d.x_test = d.x_test[:shrink]

if args.folder is None:
        args.folder = str(args.config).split(".json")[0]


os.makedirs(os.path.join('tc_final_results', args.dataset), exist_ok=True)
args.folder = os.path.join('tc_final_results', args.dataset, args.folder)
os.makedirs(args.folder, exist_ok=True)

if args.filename is None:
        args.filename = 'beta_'+str(args.beta if args.beta is not None else 1)+'_tc_'+str(args.tc_weight if args.tc_weight is not None else 0)+'_remove_'+str(args.remove_prob if 'corr' in args.dataset else 'full')+'_run_'+str(args.count)
        #+str(args.beta) if args.beta is not None
else:
   args.filename = args.filename+(str(args.count) if args.count is not None else '')+ '_beta_'+str(args.beta if args.beta is not None else 1)+'_tc_'+str(args.tc_weight if args.tc_weight is not None else 0)+('_remove_'+str(args.remove_prob) if 'corr' in args.dataset else 'full')

args.filename =os.path.join(args.folder, args.filename)

print("FILENAME ", args.filename)

dd = copy.copy(d)
m = model.NoiseModel(dd, config = config, filename = args.filename, verbose = args.verbose, seed = args.seed)
ep = m.epochs if args.epochs is None else args.epochs
try:
        ep = ep * epoch_mult
except:
        pass

m.epochs = 0 

m.fit(dd.x_train, verbose = args.verbose, validate = args.validate, fit_gen = args.fit_gen, epochs = 0)

actz = m.encoder_layers[-1]['act'][0]
actz.set_shape((m.batch, K.int_shape(actz)[1], K.int_shape(actz)[2], K.int_shape(actz)[3]))
latent = K.int_shape(m.encoder_layers[-1]['stat'][0][0])[-1]
actz.set_shape((m.batch, 1, 1, latent))





def shuffle_codes(z, batch = None):
  """Shuffles latent variables across the batch.
  Args:
    z: [batch_size, num_latent] representation.
  Returns:
    shuffled: [batch_size, num_latent] shuffled representation across the batch.
  """
  z_shuffle = []
  lim = z.get_shape()[1] #if batch is None else batch
  for i in range(lim):
    z_shuffle.append(tf.random_shuffle(z[:, i]))
  shuffled = tf.stack(z_shuffle, 1, name="latent_shuffled")
  return shuffled

def shuffle_batch(x, numpy = True, batch = 100):
  # only works for 2d / flattened tensors                                                                                                                                                                                                                   
  if isinstance(x, np.ndarray):
    x = K.variable(x)
    x = K.Flatten()(x)
  try:
          batch = x.get_shape().as_list()[0] #K.int_shape(x)[0]                                                                 
  except:
          batch = batch

  mm = x.get_shape().as_list()[-1] #K.int_shape(x)[-1]                                                                                                                                                                                                      
  #if batch is None:
  #  batch = batch_size

  def shuffle_with_return(y):
    zz = copy.copy(y)
    if len(np.shape(zz)) > 1:
        for i in range(zz.shape[-1]):
            np.random.shuffle(zz[:,i])
    else:
        np.random.shuffle(zz)
    return zz

  perm_matrix = np.array([[[row, j] for row in shuffle_with_return(np.arange(batch))] for j in range(mm)])

  return tf.transpose(tf.gather_nd(x, perm_matrix))

                        
if args.epochs is not None:
        m.epochs = int(args.epochs)


z_act = tf.squeeze(m.encoder_layers[-1]['act'][0])


z_shuff = shuffle_codes(z_act)#, batch = m.batch)


lyrs = 6
layer_size = 1000
discrim_layers = []

for _i in range(lyrs):
        discrim_layers.append(keras.layers.Dense(layer_size, activation=tf.nn.leaky_relu, name="discrim_"+str(_i)))#(lyr)#(self.inp)

logits = keras.layers.Dense(2, activation=None, name="discrim_logits")#(lyr)
probs = keras.layers.Lambda(lambda x: tf.clip_by_value(tf.nn.softmax(x), 1e-6, 1-1e-6), name = 'discrim_prob')#(logits) #(logits)
#clipped = tf.clip_by_value(probs, 1e-6, 1 - 1e-6)
discrim_layers.append(logits)
discrim_layers.append(probs)

def call_discrim(_z, _lyrs):
        t = _z
        for l in _lyrs:
              t = l(t)
              if 'logits' in l.name:
                logit = t
        return logit, t # probs is last one
                      
z_logits, z_prob = call_discrim(z_act, discrim_layers)
z_shuff_logits, z_shuff_prob = call_discrim(z_shuff, discrim_layers)

tc =  z_logits[:, 0] - z_logits[:, 1]
tc_weight = tf.placeholder(tf.float32, shape=())
tc = tf.multiply(tc_weight, tf.reduce_mean(tc, axis=0), name = 'tc_weighted_loss') #, keepdims = True) 

discrim_loss = -tf.reduce_mean(tf.add(                                                                   
        0.5 * K.log(z_prob[:, 0]),#, keepdims = True)
        0.5 * K.log(z_shuff_prob[:, 1])),# keepdims = True
                               name="discriminator_loss", axis = 0)                                                                                                                                                                                                                           



def train(m,x_train, x_val = None, y_train = None, y_val = None, anneal_ep = 0, begin_vars = [],
      min_lagr = 0.0, max_lagr = 100.0, lr_lagr = None, discrim_vars = None, sess = None,
      addl_losses = [], discrim_loss = None):
      

      #begin_vars = set(tf.global_variables())
      sess = m.sess if sess is None else sess
      def is_reg(_name):
              return ('reg' in _name or 'vae' in _name or 'echo' in _name)

      #total_loss = tf.add_n(addl_losses + [m.model_loss_weights[i]*tf.reduce_mean(losses.loss_val(m.model.outputs[i]), name = m.model.outputs[i].name.split("/")[0]+'_loss') for i in range(len(m.model.outputs))])
      model_losses = [tf.reduce_mean((args.beta if is_reg(m.model.outputs[i].name) else 1.0)*losses.loss_val(m.model.outputs[i]), name = m.model.outputs[i].name.split("/")[0]+'_loss') for i in range(len(m.model.outputs))] 
      total_loss = tf.add_n(addl_losses + model_losses)
      learning_rate = tf.placeholder(tf.float32)

      # TO DO : facilitate passing optimizer_params                                                                                                                                                                                                             
      if "Adam" in m.optimizer_name:
              trainer = tf.train.AdamOptimizer(learning_rate, **m.optimizer_params).minimize(total_loss, var_list =[v for v in tf.trainable_variables() if not 'discrim' in v.name])
      elif "SGD" in m.optimizer_name:
              trainer = tf.train.GradientDescentOptimizer(learning_rate, **m.optimizer_params).minimize(total_loss, var_list = [v for v in tf.trainable_variables() if not 'discrim' in v.name])
      else:
              trainer = tf.train.AdamOptimizer(learning_rate, **m.optimizer_params).minimize(total_loss, var_list = [v for v in tf.trainable_variables() if not 'discrim' in v.name])


      discrim_vars = [v for v in tf.trainable_variables() if 'discrim' in v.name]
      #if discrim_vars is not None:
              #if discrim_vars:
              #        discrim_trainer = tf.train.GradientDescentOptimizer(learning_rate, **m.optimizer_params).minimize(discrim_loss, var_list=discrim_vars)
              #else:
      discrim_trainer = tf.train.AdamOptimizer(learning_rate, beta1= 0.5).minimize(discrim_loss, var_list=discrim_vars)

      outputs = [losses.loss_val(out_i) for out_i in m.model_outputs] + addl_losses + [discrim_loss]
      
      to_run = [trainer, total_loss]+ outputs+ [discrim_trainer]
      to_run = to_run + (constraints.lagrange_multipliers if m.constraints is not None else [])
      #to_run = to_run + [discrim_trainer] if discrim_vars is not None else []
      tl_ind = 1

      print("Ops To Run : ", to_run)
      sess.run(tf.global_variables_initializer())
      x_train = d.x_train
      n_samples = x_train.shape[0]
      m.hist = defaultdict(list)
      print("Beginning TRAINING ", n_samples, x_train.shape)
      #with m.sess.as_default():
      if True:
        for epoch in range(int(ep)):
          print("Training Epoch ", epoch)
          epoch_avg = defaultdict(list)
                                                                                                                                                                                                                             
          perm = np.random.permutation(n_samples)
          
          if m.anneal_functions is not None:
                  for a in range(len(m.anneal_functions)):
                          try:
                                  ind = m.anneal_functions[a]['loss']
                          except:
                                  ind = m.anneal_functions[a]['ind']
                  anneal =  tf.assign(m.model_loss_weights[ind], m.anneal_functions[a]['function'](epoch))
                  m.sess.run(anneal)

          for offset in range(0, (int(n_samples / m.batch) * m.batch), m.batch):
                  batch_data = x_train[perm[offset:(offset + m.batch)]]

                  fd ={m.input_tensor: batch_data,
                       learning_rate: 0.0001 if not m.lr_callback else m.lr(epoch),
                       tc_weight: args.tc_weight}


                  
                  result = m.sess.run(to_run, feed_dict = fd)
                  
                  if epoch == 1 and offset == 0:
                          try:
                                  _act, _shuff = m.sess.run([z_act, z_shuff], feed_dict = fd)
                          except Exception as e:
                                  print(e)
                   
                   
        
                  epoch_avg['total_loss'].append(np.mean(result[tl_ind]))
        
                  for loss_ind in range(len(outputs)):
                          batch_loss = result[loss_ind+tl_ind+1]
        
                          epoch_avg[outputs[loss_ind].name].append(batch_loss)
        

          if args.validate:
                  for offset in range(0, (int(x_val.shape[0] / m.batch) * m.batch), m.batch):
                          batch_data = x_val[perm[offset:(offset + m.batch)]]
                          
                          fd ={m.input_tensor: batch_data,
                               learning_rate: m.lr if not m.lr_callback else m.lr(epoch),
                               tc_weight: args.tc_weight if not (ep == 0 and int(offset/m.batch)< anneal_ep) else 0}
                  
                          result = m.sess.run(to_run, feed_dict = fd)
                          
                          epoch_avg['val_total_loss'].append(np.mean(result[tl_ind]))
        
                          for loss_ind in range(len(outputs)):
                                  batch_loss = result[loss_ind+tl_ind+1]
                                  #batch_loss = losses.loss_val(batch_loss)
                                  epoch_avg['val_'+outputs[loss_ind].name].append(batch_loss)
                                                                                                                

          # print and record epoch averages                                                                                                                                                                                                        
          print("Epoch ", str(epoch), ": ", end ="")
          for loss_layer in epoch_avg.keys():#m.model.outputs:                                                                                                                                                                                   
                  if epoch_avg[loss_layer] is not None and epoch_avg[loss_layer][-1] is not None:
                          epoch_loss = np.mean(epoch_avg[loss_layer])# if 'echo_noise_var' not in loss_layer else np.var(epoch_avg[loss_layer])
                          m.hist[loss_layer].append(epoch_loss)
                  if True:#m.verbose:
                          print(loss_layer.split("/")[0]+" : "+str(epoch_loss)+" \t ", end="")

          if m.verbose:
                  try:
                          print(" Lagrange Multipliers : ", m.sess.run(constraints.lagrange_multipliers, feed_dict = fd))
                  except:
                          pass
          print()


import sys
orig_stdout = sys.stdout
#f_out = open(args.filename+'_log.out', 'w')
#sys.stdout = f_out

sess = m.sess
train(m, d.x_train, d.x_test, sess = sess,
      addl_losses = [tc], discrim_loss = discrim_loss)



try:
  import analysis
  analysis.disent_plot(m, d = 'dsprites')
except Exception as e:
  print()
  print("EXCPETION PLOTTING ", e)
  print()
        #import IPython
  #IPython.embed()

z_fn = m.get_layer
print("Z FN ", type(z_fn))
try:
        print(z_fn.outputs)
except Exception as e:
        print("CANNOT PRINT Z fn : ", e)
print()
try:
        print(z_fn.predict())
except:
        pass



#m.model.save_weights(args.filename+'_weights.h5')

if args.evals is None:
        args.evals = ['factor', 'mig20'] #, 'factor01']#, 'factor', 'unsupervised']
#if 'dsprites' in args.dataset:
gt = dsprites.DSprites([1,2,3,4,5])

eval_funs = {'mig50': mig.compute_mig,
             'mig20': mig.compute_mig,
                         'factor':      factor_vae.compute_factor_vae,
                         'factor01':      factor_vae.compute_factor_vae,
                         'unsupervised': unsupervised_metrics.unsupervised_metrics
                        }

eval_args = {'mig50': {'batch_size': m.batch,
                     'num_train': 10000-10000%m.batch #737280 #d.x_test.shape[0]                                                                                                                                        
             },
             'mig20': {'batch_size': m.batch,
                     'num_train': 10000-10000%m.batch, # d.x_test.shape[0],                                                                                                                                             
                       'num_bins':20
             },
             'factor': {'batch_size': m.batch,
                        'prune':0.05,
                        'num_train': 10000-10000%m.batch, #int(737280*.6), #d.x_test.shape[0],                                
                        'num_eval': 5000-5000%m.batch, #int(737280*.4), #int(d.x_test.shape[0]/2),
                        'num_variance_estimate': 10000-10000%m.batch #d.x_test.shape[0]                                                                                                                                   
                },
             'factor01': {'batch_size': m.batch,
                        'prune':0.01,
                        'num_train': 10000-10000%m.batch, #int(737280*.6), #d.x_test.shape[0],                                
                        'num_eval': 5000-5000%m.batch, #int(737280*.4), #int(d.x_test.shape[0]/2),
                        'num_variance_estimate': 10000-10000%m.batch #d.x_test.shape[0]                                                                                                                                   
                },
             'unsupervised':{'batch_size': m.batch,
                             'num_train': 10000-10000%m.batch
                     }
                        }
# eval_args = {'mig50': {'batch_size': m.batch,
#                      'num_train': 737280 #d.x_test.shape[0]                                                                                            
#              },
#              'mig20': {'batch_size': m.batch,
#                      'num_train': 737280, # d.x_test.shape[0],                                                                                         
#                        'num_bins':20
#              },

#              'factor': {'batch_size': m.batch,
#                         'num_train': 10000, #int(737280*.6), #d.x_test.shape[0],                                                                               
#                         'num_eval': 5000, #int(737280*.4), #int(d.x_test.shape[0]/2),                                                                         
#                         'num_variance_estimate': d.x_test.shape[0]
#                 },
#              'unsupervised':{'batch_size': m.batch,
#                              'num_train': 737280 #d.x_test.shape[0]                                                                                    
#                      }
#                         }

scores = {}
for method in args.evals:
        print("Evaluating ", method)
        eval_fun = eval_funs[method]
        def fn_with_args(fun, ground_truth = None,  
                                          representation_function = None,
                                          method = 'mig'):
                np_rs = np.random.RandomState(args.seed) #np.random.get_state()
                return fun(ground_truth, 
                           representation_function, 
                           np_rs,
                           **eval_args[method] 
                )
        #try:
        #        print(eval_args[method])
        #except Exception as e:
        #        print("EVAL ARGS exception : ", e)
        try:
                scores[method]= fn_with_args(eval_fun,
                                     ground_truth = gt,
                                     representation_function = z_fn, 
                                     method = method)
        except Exception as e:
                print()
                print("*"*30)
                print("Cannot calculate ", method)
                print(e)
                print("*"*30)
                print()
                continue
                # model function
                # ground truth params
        print()
        print('Method: ', method)
        print(scores[method])

try:
        for k in m.hist.keys():
                if isinstance(m.hist[k],list):
                        scores[k] = m.hist[k][-1]
                else:
                        scores[k] = m.hist[k]
except Exception as e:
        print("*"*100)
        print("Cannot get model history : ", e) 
        print("*"*100)
        try:
                scores['hist'] = m.hist
        except Exception as e:
                print("Not saving hist in full: ", e)

with open(args.filename+'_disentanglement.pickle', 'wb') as f:
        pickle.dump(scores, f)


#with open(args.filename.split("/")[-1]+'_disentanglement.pickle', 'wb') as f:
#        pickle.dump(scores, f)


m.model.save_weights(args.filename+'_weights.h5')

sys.stdout = orig_stdout
f_out.close()
