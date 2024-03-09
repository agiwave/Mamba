
import aka.boot as boot
import tensorflow.keras.datasets as datasets

from aka.providers.tf.datasets.CIFAR10 import Dataset as CIFAR10
from aka.providers.tf.datasets.CIFAR100 import Dataset as CIFAR100
from aka.providers.tf.datasets.MNIST import Dataset as MNIST
from aka.providers.tf.datasets.FashionMNIST import Dataset as FashionMNIST
from aka.providers.tf.datasets.ImageFolder import Dataset as ImageFolder

# 工厂函数
def create_object(module_name, type_name, *args, **kwargs):
    if(type_name in globals().keys()):
        return globals()[type_name](*args, **kwargs)
    
    type_name = type_name.lower()
    if(not hasattr(datasets, type_name)):
        raise TypeError("Type not implement: "+type_name)

    return getattr(datasets, type_name)(*args, **kwargs)
