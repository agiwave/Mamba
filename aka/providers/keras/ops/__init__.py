import keras.functional
import keras

# 工厂函数
def create_object(module_name, type_name, *args, **kwargs):
    if(type_name in globals().keys()):
        return globals()[type_name]
    
    type_name = type_name.lower()
    if(not hasattr(keras, type_name)):
        if(not hasattr(functional, type_name)):
            raise TypeError("Type not implement: "+type_name)
        return getattr(functional, type_name)

    return getattr(keras, type_name)

def iden(x, *args):
    if(len(args)==0):
        return x
    else:
        return x, *args

