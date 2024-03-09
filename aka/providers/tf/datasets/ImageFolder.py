import tensorflow.keras as keras
import pathlib

#
# 创建数据集
#
def Dataset(root, resize=224):

    train_ds = keras.utils.image_dataset_from_directory(
        directory=pathlib.Path(root+"/train"),
        labels='inferred',
        label_mode='int',
        batch_size=32,
        image_size=(resize, resize))

    val_ds = keras.utils.image_dataset_from_directory(
        directory=pathlib.Path(root+"/val"),
        labels='inferred',
        label_mode='int',
        batch_size=32,
        image_size=(resize, resize))
    
    return (train_ds, None), (val_ds, None)
