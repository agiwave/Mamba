from .. import boot

def load_weights(model, filename=None):return boot.invoke()
def save_weights(model, filename=None):return boot.invoke()

def train(
    model, 
    dataset, 
    *,
    data_loader=None,
    input_targets=False, 
    persist_filename = None,
    show_chart = False,
    **kwargs):return boot.invoke()

def Trainer(
    model,
    data_loader,
    input_targets=False,
    loss="CrossEntropyLoss",
    optimizer="Adam",
    optimizer_kwargs={},
    epochs = 1): return boot.invoke()

boot.inject()
