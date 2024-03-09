from .. import boot

# Shape
def Flatten(*args, **kwargs):return boot.invoke()
def Reshape(*args, **kwargs):return boot.invoke()
def Permute(*args, **kwargs):return boot.invoke()

# ImageShape
def ImageAxisC():return boot.invoke()
def ImageAxisH():return boot.invoke()
def ImageAxisW():return boot.invoke()
def ImageShape(c,w,h): return boot.invoke()

boot.inject()