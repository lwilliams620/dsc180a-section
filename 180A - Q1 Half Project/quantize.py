import tensorflow as tf

from larq import utils
from larq.layers_base import (
    QuantizerBase
)

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