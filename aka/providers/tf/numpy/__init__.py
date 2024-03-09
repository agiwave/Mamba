from tensorflow import *
import tensorflow as tf

sub = subtract
mul = multiply
div = divide
dot = tensordot
permute = transpose
def iden(*inputs):
    if(len(inputs)>1):
        return inputs
    else:
        return inputs[0]

def unfold(data, kernel_size, stride=1, padding=0 ):
    padding = 'SAME' if padding != 0 else 'VALID'
    return tf.image.extract_patches(data, sizes=[1, kernel_size, kernel_size, 1], strides=[1, stride,stride,1], rates=[1, 1, 1, 1], padding=padding)
