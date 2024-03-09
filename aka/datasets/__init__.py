from .. import boot

# Datasets
def CIFAR10(*args, **kwargs):return boot.invoke()
def CIFAR100(*args, **kwargs):return boot.invoke()
def MNIST(*args, **kwargs):return boot.invoke()
def FashionMNIST(*args, **kwargs):return boot.invoke()
def ImageFolder(*args, **kwargs):return boot.invoke()

boot.inject()