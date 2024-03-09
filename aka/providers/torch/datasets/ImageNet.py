
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from matplotlib import pyplot as plt

#
# 创建数据集
#
def Dataset(resize=224):
    transform = transforms.Compose([
        # 重新改变大小为`size`，若：height>width`,则：(size*height/width, size)
        # transforms.Resize(resize),        
        # 随机切再resize成给定的size大小          
        # transforms.CenterCrop(resize),
        transforms.RandomResizedCrop(128),
        # 把一个取值范围是[0,255]或者shape为(H,W,C)的numpy.ndarray， 
        transforms.ToTensor(),   
        # 转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor             
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),    
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))                
    ])

    train_dataset = datasets.ImageFolder("./datasets/ImageNet/train", transform)
    test_dataset = datasets.ImageFolder("./datasets/ImageNet/val", transform)

    # print(train_dataset)
    # fig = plt.figure()
    # for i in range(12):
    #     plt.subplot(3, 4, i+1)
    #     plt.tight_layout()
    #     plt.imshow(train_dataset.data[i])
    #     plt.title("Labels: {}".format(train_dataset.targets[i]))
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.show()
    return train_dataset, test_dataset