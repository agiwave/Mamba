

import tensorflow as tf
from matplotlib import pyplot as plt
import aka.providers.tf as provider

#
# 创建训练器
#
def createTrainer(model, loss="CrossEntropyLoss", optimizer="Adam", **args):
    # 更为原生的训练实现函数
    class Trainer():
        def __init__(self, model, loss, optimizer, **args):
            self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
            self.optimizer = tf.keras.optimizers.Adam()

        def train(self, dataset, epochs=5, batch_size=32, n_batch_print=100):
            (x_train, y_train) = dataset
            if(y_train is None):
                train_ds = x_train
            else:
                train_ds = tf.data.Dataset.from_tensor_slices(
                    (x_train, y_train)).shuffle(len(x_train)).batch(batch_size)

            loss_object = self.loss_object
            optimizer = self.optimizer
            train_loss = tf.keras.metrics.Mean(name='train_loss')
            train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
            batch_loss = tf.keras.metrics.Mean(name='train_loss')
            batch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
            losses, acces = [], []

            print('|'.join(format(h, f'^{12}') for h in ['Progress', 'Epoch', 'Batch', 'Batch Time', 'Train Loss', 'Train Acc']))
            print('|'.join('-'*12 for i in range(6)))
            for epoch in range(epochs):
                # 在下一个epoch开始时，重置评估指标
                train_loss.reset_states()
                train_accuracy.reset_states()
                n_batchs = 0
                for images, labels in train_ds:
                    batch_loss.reset_states()
                    batch_accuracy.reset_states()
                    with tf.GradientTape() as tape:
                        predictions = model(images)
                        predictions = tf.keras.activations.softmax(predictions)
                        loss = loss_object(labels, predictions)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                    n_batchs += 1
                    train_loss(loss)
                    train_accuracy(labels, predictions)
                    batch_loss(loss)
                    batch_accuracy(labels, predictions)

                    print('|'.join(format(str(item), f'^{12}') for item in [
                        '?', # {:.2f}%'.format(progress),
                        epoch+1,
                        n_batchs,
                        '?', # {:.3f}s'.format(batch_time), 
                        '{:.4f}'.format(batch_loss.result()),
                        '{:.4f}'.format(batch_accuracy.result())
                    ]), end='\r')

                    if(n_batchs%n_batch_print == 0):
                        losses.append(train_loss.result())
                        acces.append(train_accuracy.result())
                        train_loss.reset_states()
                        train_accuracy.reset_states()
                print('')
            return losses, acces

        def evaluate(self, dataset, batch_size=1):
            (x_train, y_train) = dataset
            train_ds = tf.data.Dataset.from_tensor_slices(
                (x_train, y_train)).batch(batch_size)

            loss_object = self.loss_object
            train_loss = tf.keras.metrics.Mean(name='test_loss')
            train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
            train_loss.reset_states()
            train_accuracy.reset_states()
            for images, labels in train_ds:
                with tf.GradientTape() as tape:
                    predictions = model(images)
                    loss = loss_object(labels, predictions)
                gradients = tape.gradient(loss, model.trainable_variables)
                train_loss(loss)
                train_accuracy(labels, predictions)

            template = 'Test Loss: {}, Test Accuracy: {}'
            print (template.format(train_loss.result(),
                                    train_accuracy.result()))
            return train_loss.result(),train_accuracy.result()

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