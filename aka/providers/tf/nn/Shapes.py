
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

def ImageAxisC():return 2
def ImageAxisH():return 0
def ImageAxisW():return 1
def ImageShape(c,w,h) :
    return [w, h, c]

def Reshape(*args):
    return layers.Reshape(args)

def Permute(*args):
    return layers.Permute(args)

def Flatten():
    return layers.Flatten()
