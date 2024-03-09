

import os
import importlib
from matplotlib import pyplot as plt
import aka.providers.keras as provider

#
# 创建训练器
#
def createTrainer(model, loss="CrossEntropyLoss", optimizer="Adam", **args):
    class Trainer():
        def __init__(self, model, loss, optimizer, **args):
            model.compile(optimizer=optimizer.lower(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
            self.model = model

        def train(self, dataset, epochs=5, batch_size=32):
            (inputs, targets) = dataset
            if(targets == None):
                batch_size = None
            r = self.model.fit(
                x=inputs,
                y=targets,
                epochs=epochs,
                batch_size=batch_size
            )
            return r.history["loss"], r.history["accuracy"]

        def evaluate(self, dataset, batch_size=1):
            (inputs, targets) = dataset
            return self.model.evaluate(inputs, targets)

    return Trainer(model=model, loss=loss, optimizer=optimizer, **args)


def toprovider(data) :
    if(hasattr(data, "convert_to_provider")) :
        return getattr(data, "convert_to_provider")(provider)
    return data

#
# 训练模型
#
def train(model, dataset, **kwargs):

    # 模型
    train_model = toprovider(model)
    train_model.summary()

    # 数据集
    train_set, test_set = toprovider(dataset)

    # 训练器
    trainer = createTrainer(model=train_model, **kwargs)

    # 训练
    losses, acces = trainer.train(train_set, **kwargs)

    #
    # 图像化输出训练结果
    #
    plt.plot(losses)
    plt.plot(acces)
    plt.xlabel('Iterators')
    plt.ylabel('Loss & Acc')
    plt.show()
    return losses, acces