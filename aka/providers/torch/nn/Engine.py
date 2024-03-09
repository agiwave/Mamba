import os
import time
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchinfo import summary

def load_weights(model, filename):
    model.load_state_dict(torch.load(filename))
    return model
    
def save_weights(model, filename):
    torch.save(model.state_dict(), filename)
    return model

def load(filename):
    return torch.load(filename)

def save(model, filename):
    return torch.save(model, filename)

def train(model, datasets=None, data_loader=None, 
        persist_filename = None, 
        epochs = 1, batch_size=64, 
        show_chart = False, **kwargs):
    # 模型
    if(persist_filename!=None and os.path.exists(persist_filename)):
        load_weights(model, persist_filename)
    summary(model)

    # -- data loader --
    if(data_loader is None):
        train_set, _ = datasets
        data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    trainer = Trainer(model, data_loader=data_loader, epochs=epochs, batch_size=batch_size, **kwargs)
    train_losses = []
    train_acces = []
    a_acces = []
    a_losses = []
    a_loss = 0.
    a_acc = 0.
    last_print_time = time.time()
    n_losses = 1
    n_batchs = 0
    header_printed = False
    last_print_epchs = 0
    for ctx in trainer:
        if(ctx.i_epoch>=0):
            # -- Print headers --
            if(header_printed==False):
                header_printed = True
                print_headers = ['Progress', 'Epoch', 'Batch', 'Batch Time']
                if(isinstance(ctx.outputs,tuple)):
                    losses = ctx.outputs[-1]
                    if(isinstance(losses,tuple)):
                        n_losses = len(losses)
                        a_losses.append(0.0)
                        print_headers += ['loss:' + str(i) for i in range(len(losses))]
                    else:
                        print_headers += ['loss', 'acc']
                else:
                    print_headers += ['loss', 'acc']
                print('='*(64+13*n_losses))
                print('|'.join(format(h, f'^{12}') for h in print_headers))
                print('|'.join('-'*12 for i in range(len(print_headers))))

            if(last_print_epchs != ctx.i_epoch):
                last_print_epchs = ctx.i_epoch
                # -- Persist --
                if(persist_filename!=None):
                    save_weights(model, persist_filename)
                print('')
            
            # -- Print batch result if passed time over 0.2s --
            curr_time = time.time()
            if(curr_time-last_print_time > 0.2):
                batch_time = ctx.batch_time
                progress = (ctx.i_batchs+1)*100.0/ctx.n_batchs
                if(n_losses==1):
                    print('|'.join(format(str(item), f'^{12}') for item in [
                        '{:.2f}%'.format(progress),
                        ctx.i_epoch+1,
                        ctx.i_batchs+1,
                        '{:.3f}s'.format(batch_time), 
                        '{:.4f}'.format(ctx.loss),
                        '{:.4f}'.format(ctx.acc)
                    ]), end='\r')
                else:
                    print('|'.join(format(str(item), f'^{12}') for item in [
                        '{:.2f}%'.format(progress),
                        ctx.i_epoch+1,
                        ctx.i_batchs+1,
                        '{:.3f}s'.format(batch_time), 
                        *['{:.4f}'.format(loss.item()) for (_, loss) in losses]
                    ]), end='\r')
                last_print_time = curr_time

            n_batchs += 1
            n_batch_print = ctx.n_batchs*epochs//100

            # -- Acc losses --
            if(n_losses==1):
                a_loss += ctx.loss/n_batch_print
                a_acc += ctx.acc/n_batch_print
            else:
                for i in range(n_losses):
                    a_losses[i] += ctx.losses[i]/n_batch_print

            # -- Log Losses --
            if(n_batchs % n_batch_print == 0):
                if(n_losses == 1):
                    train_losses.append(a_loss)
                    train_acces.append(a_acc)
                    a_loss = 0.0
                    a_acc = 0.0
                else:
                    for i in range(n_losses):
                        train_losses[i][1].append(a_losses[i])
                        a_losses[i] = 0.0

    print('\n'+'='*(64+13*n_losses))

    # -- Persist --
    if(persist_filename!=None):
        save_weights(model, persist_filename)

        # 图像化输出训练结果
    if(show_chart and ctx.train_mode!=0):
        if(ctx.train_mode==3):
            keys = []
            for i in range(len(train_losses)):
                keys.append('loss:'+str(i))
                plt.plot(train_losses[i][1])
            plt.xlabel('Iterators')
            plt.ylabel('Losses')
            plt.legend(keys, loc='upper right')
            plt.show()
        else:
            plt.plot(train_losses)
            plt.plot(train_acces)
            plt.xlabel('Iterators')
            plt.ylabel('Loss & Acc')
            plt.legend(['loss', 'acc'], loc='upper right')
            plt.show()

    return train_losses

class TrainArgs:
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

def Trainer(
    model,
    data_loader = None,
    loss="CrossEntropyLoss",
    input_targets=False,
    optimizer="Adam",
    optimizer_kwargs={},
    epochs=2,
    **kwargs):

    # -- Train Variables --
    train_mode = 0  # 0 -- Uninitialized, 1 -- Without loss, 2 -- With loss --, 3 -- Dict loss --
    ctx = TrainArgs(
        train_mode = 0,
        n_batchs = len(data_loader),
        i_epoch = 0,
        i_batch = 0,
        inputs = None,
        targets = None,
        outputs = None,
        losses = [],
        loss = 0.,
        acc = 0.
    )

    # -- epochs --
    for ctx.i_epoch in range(epochs):
        # -- iter --
        ctx.i_batchs = -1
        for (ctx.inputs, ctx.targets) in data_loader:
            ctx.i_batchs += 1
            start_time = time.time()

            # -- Forward --
            if(input_targets):
                ctx.outputs = model(ctx.inputs, targets=ctx.targets)
            else:
                ctx.outputs = model(ctx.inputs)

            # -- Initialize at first time --
            if(ctx.train_mode == 0):
                # -- Update mode --
                if(isinstance(ctx.outputs,tuple)):
                    losses = ctx.outputs[-1]
                    if(isinstance(losses,tuple)):
                        ctx.train_mode = 3
                    else:
                        ctx.train_mode = 2
                else:
                    ctx.train_mode = 1
                    criterion = getattr(torch.nn,loss)()

                # -- Initialize --
                if(ctx.train_mode == 3):
                    ctx.train_losses, a_losses, t_optimizers = [], [], []
                    for (m, _) in losses:
                        if(m is None):
                            m = model
                        optim = getattr(torch.optim,optimizer)(m.parameters(), **optimizer_kwargs)
                        t_optimizers.append(optim) 
                else:
                    t_optimizer = getattr(torch.optim,optimizer)(model.parameters(), **optimizer_kwargs)

            # -- Backward --
            if(ctx.train_mode != 3):
                if(ctx.train_mode==2):
                    (_, loss) = ctx.outputs
                    ctx.acc = 0.0
                else:
                    loss = criterion(ctx.outputs, ctx.targets)
                    _, predicted = torch.max(ctx.outputs.data, dim = 1)
                    ctx.acc = (predicted == ctx.targets).sum().item() * 1.0 / ctx.targets.size(0)
                    
                t_optimizer.zero_grad()
                loss.backward()
                t_optimizer.step()
                ctx.loss = loss.item()
                curr_time = time.time()
                ctx.batch_time = curr_time-start_time
                yield ctx
            else:
                # -- Optimize --
                (_, losses) = ctx.outputs
                i=0
                ctx.losses = []
                for (_, loss) in losses:
                    optim = t_optimizers[i]
                    optim.zero_grad()
                    retain_graph = True if i != n_losses-1 else False
                    loss.backward(retain_graph=retain_graph)
                    ctx.losses.append(loss.item())
                    i+=1
                for optim in t_optimizers:
                    optim.step()

                curr_time = time.time()
                ctx.batch_time = curr_time-start_time
                yield ctx