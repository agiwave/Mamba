
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from matplotlib import pyplot as plt

#
# 创建数据集
#
def Dataset(root="data", download=True, **args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.1307,),std = (0.3081,))
    ])

    train_dataset = datasets.FashionMNIST(root=root, download=download, train=True, transform=transform)  # 本地没有就加上download=True
    test_dataset = datasets.FashionMNIST(root=root, download=download, train=False, transform=transform)  # train=True训练集，=False测试集

    fig = plt.figure()
    for i in range(12):
        plt.subplot(3, 4, i+1)
        plt.tight_layout()
        plt.imshow(train_dataset.data[i])
        plt.title("Labels: {}".format(train_dataset.targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()
    return train_dataset, test_dataset