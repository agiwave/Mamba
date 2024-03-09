import aka.boot
import keras as provider

from .engine import train

# 工厂函数
def create_object(module_name, type_name, *args, **kwargs):
    return aka.boot.create_object_in_provider(module_name, type_name, *args, **kwargs)




