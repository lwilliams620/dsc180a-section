import tensorflow as tf

import larq
from larq.layers_base import QuantizerBase
from larq import context, math, utils
import numpy as np

@utils.register_keras_custom_object
class QuantLSTM(QuantizerBase, tf.keras.layers.LSTM):
     def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        input_quantizer=None,
        kernel_quantizer=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            units,
            activation=activation,
            use_bias=use_bias,
            input_quantizer=input_quantizer,
            kernel_quantizer=kernel_quantizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

def hard_sigmoid(x):
    return tf.clip_by_value((x + 1.)/2., 0, 1)
    
def make_quantize(x, a, n): # x: weight vector, a: maximum weight value, n: number of quantization points (3bits, n=8)
    temp = tf.math.round(((x+a)/2) * (n-1)/(a)) * a/(n-1) * 2 - a
    
    mask = tf.greater_equal(tf.math.abs(temp), a)
    temp = tf.where(mask, tf.math.sign(temp)*a, temp)
    return temp

def ste_sign(x: tf.Tensor, clip_value) -> tf.Tensor:
    @tf.custom_gradient
    def _call(x):
        def grad(dy):
            return larq.quantizers._clipped_gradient(x, dy, clip_value[0])
        
        temp = make_quantize(x, clip_value[0], clip_value[1])

        return temp, grad

    return _call(x)