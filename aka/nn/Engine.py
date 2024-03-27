from .. import boot

def load_weights(model, filename=None):return boot.invoke()
def save_weights(model, filename=None):return boot.invoke()

def train(
    model, 
    datasets = None, 
    *,
    data_loader = None,
    persist_filename = None,
    show_chart = False,
    **kwargs):return boot.invoke()

def Trainer(
    model,
    data_loader,
    data_fields = None,
    loss_metric = None,
    optimizer="Adam",
    optimizer_kwargs={},
    epochs = 1): return boot.invoke()

boot.inject()
