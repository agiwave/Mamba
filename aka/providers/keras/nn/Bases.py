
import keras.funcs as funcs

class Parameter:
    def __init__(shape, initializer, requires_grad):
        self.shape = shape
        self.trainable = requires_grad
        self.initializer = initializer

def Input(input_shape):
    return layers.Input(shape=input_shape)
