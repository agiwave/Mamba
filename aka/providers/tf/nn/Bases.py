
import tensorflow as tf
import tensorflow.keras.layers as layers

def get_initializer(initializer) :
    match initializer:
        case "zeros":
            return tf.zeros_initializer()
        case "randn":
            return tf.random_normal_initializer()
        case "randu":
            return tf.random_uniform_initializer()
        case "ones":
            return tf.ones_initializer()
        case _:
            return tf.random_uniform_initializer()

def Parameter(data=None, shape=None, requires_grad=True, initializer="zeros", **kwargs):
    if(data == None) :
        data = get_initializer(initializer)(shape)
    return tf.Variable(
        initial_value = data,
        trainable = requires_grad
    )

def Input(input_shape):
    return layers.Input(shape=input_shape)
