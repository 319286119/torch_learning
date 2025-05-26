import torch.optim
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

my_layer=My_Layers()
my_layer.load_state_dict(torch.load("trained_model.pth"))
step=1
writer=SummaryWriter(log_dir="log")
loss=nn.CrossEntropyLoss()

optim=torch.optim.SGD(params=my_layer.parameters(),lr=0.01)

for i in range(20):
    total_loss=0
    for data in dataloader:
        img,target=data
        output=my_layer(img)
        loss_ret=loss(output,target)
        optim.zero_grad()
        loss_ret.backward()
        optim.step()
        total_loss+=loss_ret
    print(total_loss)

torch.save(my_layer.state_dict(), 'trained_model.pth')#保存训练的参数
