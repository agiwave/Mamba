import os
import time
import torch
import torchvision
import aka.numpy as np

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchinfo import summary

def load_weights(model, filename):
    model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
    return model
    
def save_weights(model, filename):
    torch.save(model.state_dict(), filename)
    return model

def train(
        model, 
        datasets=None,
        *,
        data_loader = None,
        collate_fn=None,
        persist_filename = None, 
        persist_per_batchs = None,
        shuffle = True,
        batch_size = 8,
        show_frequency = 0.2,
        epochs = 1,
        show_chart = False, **kwargs):

    # 模型
    if(persist_filename!=None and os.path.exists(persist_filename)):
        load_weights(model, persist_filename)
    summary(model)

    '''
    The default collate_fn conversion:
    Element Type:
        int, float, ... -> Tensor
        Dict, List, Tuple ... -> Dict(Tensor), List(Tensor), Tuple(Tensor)  (Keep Structure)
    We need:
        ?????? How to support huggingface datasets, Only dataset_collate_fn ???????
    '''
    # -- data loader --
    if data_loader is None:
        data_loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    # -- trainer --
    trainer = Trainer(model, data_loader, epochs=epochs, **kwargs)
    train_losses = []
    train_acces = []
    a_losses = []
    last_print_time = time.time()
    n_batchs = 0
    header_printed = False
    last_print_epchs = 0
    for ctx in trainer:
        if(ctx.i_epoch>=0):
            # -- Print headers --
            if(header_printed==False):
                n_losses = len(ctx.losses)
                a_losses = [0.0 for _ in range(n_losses)]
                train_losses = [[] for _ in range(n_losses)]
                header_printed = True
                print_headers = ['Progress', 'Epoch', 'Batch', 'Batch Time']
                print_headers += ['loss:' + str(i) for i in range(n_losses)]
                print('='*(64+13*n_losses))
                print('|'.join(format(h, f'^{12}') for h in print_headers))
                print('|'.join('-'*12 for i in range(len(print_headers))))
                print('')

            if(last_print_epchs != ctx.i_epoch):
                last_print_epchs = ctx.i_epoch
                # -- Persist --
                if(persist_filename!=None):
                    save_weights(model, persist_filename)
                # print('')
            
            # -- Print batch result if passed time over 0.2s --
            curr_time = time.time()
            if(curr_time-last_print_time > show_frequency):
                batch_time = ctx.batch_time
                progress = (ctx.i_batchs+1)*100.0/ctx.n_batchs
                print('\033[A\033[2K'+'|'.join(format(str(item), f'^{12}') for item in [
                    '{:.2f}%'.format(progress),
                    ctx.i_epoch+1,
                    ctx.i_batchs+1,
                    '{:.3f}s'.format(batch_time), 
                    *['{:.4f}'.format(loss) for loss in ctx.losses]
                ]))
                last_print_time = curr_time

            n_batchs += 1
            if persist_filename is not None and persist_per_batchs is not None:
                if (n_batchs % persist_per_batchs) == 0:
                    save_weights(model, persist_filename)
            n_batch_print = ctx.n_batchs*epochs//100
            if n_batch_print == 0:
                n_batch_print = 1

            # -- Acc losses --
            for i in range(n_losses):
                a_losses[i] += ctx.losses[i]/n_batch_print

            # -- Log Losses --
            if(n_batchs % n_batch_print == 0):
                for i in range(n_losses):
                    train_losses[i].append(a_losses[i])
                    a_losses[i] = 0.0

    print('\n'+'='*(64+13*n_losses))

    # -- Persist --
    if(persist_filename!=None):
        save_weights(model, persist_filename)

        # 图像化输出训练结果
    if(show_chart):
        keys = []
        for i in range(len(train_losses)):
            keys.append('loss:'+str(i))
            plt.plot(train_losses[i])
        plt.xlabel('Iterators')
        plt.ylabel('Losses')
        plt.legend(keys, loc='upper right')
        plt.show()
    if len(train_losses) == 1:
        return train_losses[0]
    else:
        return train_losses

class TrainArgs:
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
    
def Trainer(
    model,
    data_loader,
    data_fields=None,
    optimizer="Adam",
    optimizer_kwargs={},
    loss_metric = None,
    data_parallel = False,
    forward_kwargs={},
    epochs=2,
    dtype=None,
    **kwargs):

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        model = model.to(device)
        if data_parallel is True:
            model = torch.nn.DataParallel(model)
    else:
        device = torch.device("cpu")
    if dtype is not None:
        model = model.to(dtype)

    # -- Train Variables --
    train_mode = 0  # 0 -- Uninitialized, 2 -- With loss --, 3 -- Dict loss --
    ctx = TrainArgs(
        train_mode = 0,
        n_batchs = len(data_loader),
        i_epoch = 0,
        i_batch = 0,
        losses = []
    )

    # -- epochs --
    for ctx.i_epoch in range(epochs):
        # -- iter --
        ctx.i_batchs = -1
        for item in data_loader:
            ctx.i_batchs += 1
            start_time = time.time()

            # -- Forward --
            if data_fields is None:
                (inputs, targets) = item
            else:
                (inputs, targets) = item[data_fields[0]], item[data_fields[1]]

            inputs = inputs.to(device)
            targets = targets.to(device)
            if dtype is not None:
                inputs = inputs.to(dtype)
                targets = targets.to(dtype)

            if loss_metric is None:
                (outputs, loss) = model(inputs, targets=targets, **forward_kwargs)
            else:
                outputs = model(inputs)
                loss = loss_metric(outputs, targets)

            # -- Initialize at first time --
            if(ctx.train_mode == 0):
                # -- Update mode --
                if(isinstance(loss,tuple)):
                    ctx.train_mode = 3
                    ctx.tran_n_losses = len(loss)
                    ctx.train_losses, a_losses, t_optimizers = [], [], []
                    for (m, _) in loss:
                        if(m is None):
                            m = model
                        optim = getattr(torch.optim,optimizer)(m.parameters(), **optimizer_kwargs)
                        t_optimizers.append(optim) 
                else:
                    ctx.train_mode = 2
                    ctx.tran_n_losses = 1
                    t_optimizer = getattr(torch.optim,optimizer)(model.parameters(), **optimizer_kwargs)

            # -- Backward --
            if(ctx.train_mode != 3):
                t_optimizer.zero_grad()
                loss.backward()
                t_optimizer.step()
                ctx.losses = [loss.item()]
                curr_time = time.time()
                ctx.acc = 0.0
                ctx.batch_time = curr_time-start_time
                yield ctx
            else:
                # -- Optimize --
                i=0
                ctx.losses = []
                for (_, subloss) in loss:
                    optim = t_optimizers[i]
                    optim.zero_grad()
                    retain_graph = True if i != len(loss)-1 else False
                    subloss.backward(retain_graph=retain_graph)
                    ctx.losses.append(subloss.item())
                    i+=1
                for optim in t_optimizers:
                    optim.step()

                curr_time = time.time()
                ctx.batch_time = curr_time-start_time
                yield ctx