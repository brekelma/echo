import tensorflow.keras.backend as K

def tanh64(x, y = 64):
        return K.tanh(x/y)

def tanh16(x, y = 16):
        return K.tanh(x/y)

def log_sig64(x, y = 64):
        return K.log(1.0/(1+K.exp(-x/(y*1.0))))

