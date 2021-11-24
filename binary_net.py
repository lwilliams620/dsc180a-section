import  time
from collections import OrderedDict
import numpy as np
import tensorflow as tf

def hard_sigmoid(x):
    return tf.clip_by_value((x+1)/2,0,1)

def binary_tanh_unit(x):
    return 2*tf.math.round(hard_sigmoid(x))-1

def binarization(W,H):
    # [-1,1] -> [0,1]
    Wb = hard_sigmoid(W/H)
    # Wb = T.clip(W/H,-1,1)

    # Deterministic BinaryConnect (round to nearest)
    # print("det")
    Wb = tf.math.round(Wb)

    # 0 or 1 -> -1 or 1
    a = np.ones(len(Wb))
    Wb = tf.cast(tf.keras.backend.switch(Wb,tf.ones(shape=Wb.shape),tf.ones(shape=Wb.shape)*-1), tf.float32)

    return Wb

class Dense(tf.keras.layers.Dense):
    def __init__(self, units, **kwargs):
        self.H = 1
        
        super(Dense, self).__init__(units, kernel_initializer=tf.keras.initializers.RandomUniform(-self.H,self.H), **kwargs)
        self.W = self.kernel_initializer(shape=(1, 28*28))

    def call(self, input, **kwargs):
        self.Wb = binarization(self.W,self.H)
        Wr = self.W
        self.kernel_initializer = self.Wb

        rvalue = super(Dense, self).call(input)
        
        self.kernel_initializer = Wr
        
        return rvalue
