import keras.backend as K

def tanh64(x, y = 64):
        return (K.exp(1.0/y*x)-K.exp(-1.0/y*x))/(K.exp(1.0/y*x)+K.exp(-1.0/y*x)+K.epsilon())

def log_sig64(x, y = 64):
        return K.log(1.0/(1+K.exp(-x/(y*1.0))))

