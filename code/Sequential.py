import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10(root="CIFAR10",train=False,transform=torchvision.transforms.ToTensor())

dataloader=DataLoader(dataset=dataset,batch_size=64)

class My_Layers(nn.Module):
    def __init__(self):
        super().__init__()
        self.module1=nn.Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64, 10),
        )

    def forward(self,x):
        return self.module1(x)


if __name__=="__main__":
    my_layer=My_Layers()
    step=1
    writer=SummaryWriter(log_dir="log")
    loss=nn.CrossEntropyLoss()
    for data in dataloader:
        img,target=data
        writer.add_images(tag="input",img_tensor=img,global_step=step)
        writer.add_images(tag="output",img_tensor=my_layer(img),global_step=step)



