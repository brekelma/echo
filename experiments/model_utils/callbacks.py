from keras.callbacks import Callback
import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.backend import get_session
from keras.backend import eval
from collections import defaultdict

class BetaCallback(Callback):
    def __init__(self, functions, layers):#betas, sched, screening = False, recon = False): # mod, minbeta=0):
        self.layers = layers
        self.anneal_functions = functions
    
    def on_epoch_begin(self, epoch, logs={}):
        for l in range(len(self.layers)): 
            tf.assign(self.layers[l],self.anneal_functions[l](epoch)).eval(session=get_session())
            #print("anneal value ", self.layers[l])
            
class ZeroAnneal(Callback):
    def __init__(self, lw = 1, index = 0, epochs = 25, scaled = 10000000):
        self.lw = tf.constant(lw, dtype = tf.float32)
        self.zero_epochs = epochs
        self.ind = index
        self.replace = scaled*lw
        
    def on_epoch_begin(self, epoch, logs={}):
        if epoch < self.zero_epochs:
            tf.assign(self.model.loss_weights[self.ind], tf.constant(self.replace, dtype = tf.float32)).eval(session=get_session())
        else:
            tf.assign(self.model.loss_weights[self.ind],self.lw).eval(session=get_session())


class RecordEcho(Callback):
    def __init__(self):
        self.hist = []

    def on_epoch_end(self, epoch, logs={}):
        #self.hist.append(0.0)
        self.model.record_reg()
        #for l in range(len(self.model.layers)): 
        #    if 'echo' in l:
        #        self.hist.append(self.model.get_echo_dj())

class RecordVAE(Callback):
    def __init__(self):
        self.hist = []
        self.sess = K.get_session() # or passed
        self.x = tf.Variable(0., validate_shape=False)

    def on_epoch_end(self, epoch, logs={}):
        
        for loss_tensor in range(len(self.model.outputs)): 
            with self.sess.as_default():
                loss_val = self.sess.run(loss_tensor, feed_dict={})
                loss_val = tf.reduce_mean(loss_val, axis = 0)
                
        # LAGRANGIAN OPT NECESSARY TO TAKE THE AVERAGE OVER BATCH ( for each dimension )
class MyLearningRateScheduler(Callback):
    """Learning rate scheduler.                                                                                                                         
                                                                                                                                                        
    # Arguments                                                                                                                                         
        schedule: a function that takes an epoch index as input                                                                                         
            (integer, indexed from 0) and current learning rate                                                                                         
            and returns a new learning rate as output (float).                                                                                          
        verbose: int. 0: quiet, 1: update messages.                                                                                                     
    """

    def __init__(self, schedule, verbose=0):
        super(MyLearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            if not hasattr(self.model.optimizer.optimizer, 'lr'):
                raise ValueError('Optimizer must have a "lr" attribute.')
            else:
                self.lr_loc = self.model.optimizer.optimizer.lr
        else:
            self.lr_loc = self.model.optimizer.lr
                
        lr = float(K.get_value(self.lr_loc))
        try:  # new API                                                                                                                                 
            lr = self.schedule(epoch, lr)
        except TypeError:  # old API for backward compatibility                                                                                         
            lr = self.schedule(epoch)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.lr_loc, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.lr_loc)




class DensityTrain(Callback):
    def __init__(self, inputs, loss_names, outputs, losses, weights = None, lr = .0003):#betas, sched, screening = False, recon = False): # mod, minbeta=0):
        self.model = inputs
        #self.inputs = inputs
        self.loss_names = loss_names
        self.outputs = outputs
        self.losses = losses
        self.loss_weights = weights
        #self.input_layers = input_layers
        self.trainers = {}
        print("TRAINABLES ")
        print()
        #print(tf.trainable_variables())
        print()
        prev = 0
        #print(self.model.summary())
        named_vars = [v for v in tf.trainable_variables() if "masked_autoregressive" in v.name] 
        #print("NAMED VARS for loss:  ", named_vars)
        print("losses ", self.losses)
        print(self.loss_names)
        print("input ", self.model.inputs)
        #print("Input Layer: ", self.inputs)
        print()
        print('optimizer ', self.model.optimizer)
        print()
        for l in self.loss_names.keys():
            # ONLY ONE PER
            v = self.loss_names[l]
            loss = self.losses[v-1]
            print("LOSS ", loss, " called on ", self.outputs[v-1], " key ", l)
            self.trainers[l] = tf.train.AdamOptimizer(lr).minimize(loss(self.outputs[v-1])*self.loss_weights[v-1], var_list=named_vars)
            #with self.sess.as_default():
            #    self.sess.run(tf.variables_initializer(self.trainers[l].variables()))
            #self.trainers[l] = self.trainers[l].minimize(loss(self.outputs[v-1])*self.loss_weights[v-1])#, var_list=named_vars)
        self.sess = tf.Session()    
        with self.sess.as_default():
            tf.global_variables_initializer().run()
        
        self.num_batches = 0
        self.avg = {}
        self.hist = defaultdict(list)
        self.keys = list(self.trainers.keys())
        # for i in self.layer_names.keys():
        #     #learning_rate = tf.placeholder(tf.float32)
        #     named_vars = [v for v in tf.trainable_variables() if i.split("_")[0] in v.name]
        #     print("NAMED VARS for loss: ", i, " : ", named_vars)
        #     print("losses ", self.losses[prev:(self.layer_names[i]-1)])
        #     for l in self.losses[prev:(self.layer_names[i]-1)]:
        #         self.trainers[l] = tf.train.AdamOptimizer(lr).minimize(l, var_list=named_vars)
        #     prev = self.layer_names[i]

        self.x = tf.Variable(0., validate_shape=False)
        self.z = tf.Variable(0., validate_shape=False)

        #self.sess = tf.Session()
        #with self.sess.as_default():
            #for l in self.trainers.keys():
            #    self.sess.run(tf.variables_initializer(self.trainers[l].variables()))
        #    tf.global_variables_initializer().run()
        #print(self.model.trainable_weights)

    def on_epoch_end(self, epoch, logs={}):
        #print(self.avg)
        if self.avg != {}:
            for k in self.avg.keys():
                v = self.avg[k]
                self.avg[k] = self.avg[k]/self.num_batches
                try:
                    self.hist[k].extend(self.avg[k])
                except:
                    self.hist[k].append(self.avg[k])
                print("Epoch: ", epoch, ":    Loss ", k, " : ", self.avg[k]) 
                self.avg[k] = 0
        self.num_batches = 0


    def on_batch_end(self, batch, logs=None):
        # evaluate the variables and save them into lists
        x = eval(self.x)
        z = eval(self.z)

        
        try:
            self.num_batches += 1
        except:
            self.num_batches = 0
        # train 
        
        feed = {}
        for i in range(len(self.model.inputs)):
            feed[self.model.inputs[i]] = x

        with self.sess.as_default():
            #self.sess.run(self.model.optimizer, feed_dict = feed)
            self.sess.run([self.trainers[k] for k in self.trainers.keys()], feed_dict=feed)
            losses = self.sess.run([self.losses[i](self.outputs[i]) for i in range(len(self.outputs))], feed_dict=  feed)
        
        #print(losses)
        
        for i in range(len(losses)):
            try:
                self.avg[self.keys[i]] += losses[i][0]
            except:
                self.avg[self.keys[i]] = losses[i][0]
        #print(self.avg)

class DensityEpoch(Callback):
    def __init__(self, data, model, loss_names, outputs, losses,  weights = None, batch = 100, lr = .0003):#betas, sched, screening = False, recon = False): # mod, minbeta=0):
        self.data = data
        self.model = model
        self.loss_names = loss_names
        self.outputs = outputs
        self.losses = losses
        self.batch = batch
        #self.weights = weights
        self.trainers = {}
        print("TRAINABLES ")
        print()
        #print(tf.trainable_variables())
        print()
        prev = 0

        named_vars = [v for v in tf.trainable_variables() if "masked_autoregressive" in v.name] 
        #print("NAMED VARS for loss:  ", named_vars)
        print("losses ", self.losses)
        print(self.loss_names)
        for l in self.loss_names.keys():
            # ONLY ONE PER
            v = self.loss_names[l]
            loss = self.losses[v-1]
            self.trainers[l] = tf.train.AdamOptimizer(lr).minimize(loss(self.outputs[v-1]), var_list=named_vars)
        
        self.avg = {}
        self.hist = defaultdict(list)
        self.keys = list(self.trainers.keys())
        # for i in self.layer_names.keys():
        #     #learning_rate = tf.placeholder(tf.float32)
        #     named_vars = [v for v in tf.trainable_variables() if i.split("_")[0] in v.name]
        #     print("NAMED VARS for loss: ", i, " : ", named_vars)
        #     print("losses ", self.losses[prev:(self.layer_names[i]-1)])
        #     for l in self.losses[prev:(self.layer_names[i]-1)]:
        #         self.trainers[l] = tf.train.AdamOptimizer(lr).minimize(l, var_list=named_vars)
        #     prev = self.layer_names[i]

        #self.x = tf.Variable(0., validate_shape=False)
        #self.z = tf.Variable(0., validate_shape=False)

        self.sess = tf.Session()
        with self.sess.as_default():
            for l in self.trainers.keys():
                self.sess.run(variables_initializer(self.trainers[l].variables()))
            #tf.global_variables_initializer().run()


    def on_epoch_end(self, epoch, logs={}):
        #x = eval(self.x)
        #z = eval(self.z)
        
        print("x shape ", self.data.shape)
        #print("z shape ", z.shape)
        if epoch <=1 :
            self.sess = tf.Session()
            with self.sess.as_default():
                tf.global_variables_initializer().run()
        n_samples = self.data.shape[0]
        self.num_batches = int(n_samples / self.batch)
        self.sess = tf.Session()
        self.hist = defaultdict(list)
        with self.sess.as_default():
            epoch_avg = defaultdict(list)
            total_avg = []
            lagr_avg = []
            lm_avg = []
            perm = np.random.permutation(n_samples)  # random permutation of data for each epoch
            
            


            for offset in range(0, (int(n_samples / self.batch) * self.batch), self.batch):  # inner
                batch_data = self.data[perm[offset:(offset + self.batch)]]
                self.sess.run([self.trainers[k] for k in self.keys], feed_dict={self.model.inputs[0]: batch_data})
                losses = self.sess.run([self.losses[i](self.outputs[i]) for i in range(len(self.outputs))], feed_dict={self.model.inputs[0]: batch_data})
                
                for i in range(len(losses)):
                    try:
                        self.avg[self.keys[i]] += losses[i]
                    except:
                        self.avg[self.keys[i]] = losses[i]
                
        print("Epoch: ", epoch, ":    Losses ", [(k, self.avg[k]/self.num_batches) for k in self.avg.keys()])
        for k, v in self.avg:
            self.avg[k] = self.avg[k]/self.num_batches
            self.hist[k].append(self.avg[k]) 
            self.avg[k] = 0
        
