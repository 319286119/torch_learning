import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("CIFAR10",train=False,transform=torchvision.transforms.ToTensor())

dataloader=DataLoader(dataset,batch_size=64)

class My_conv(nn.Module):
    def __init__(self):
        super(My_conv,self).__init__()
        self.conv=Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
    def forward(self,x):
        x=self.conv(x)
        return x

'''my_conv=My_conv()
print(my_conv)'''

writer=SummaryWriter("log")
step=1
my_conv=My_conv()
for data in dataloader:
    imgs,target=data
    output=my_conv(imgs)
    print(imgs.shape)
    print(output.shape)
    output=torch.reshape(output,[-1,3,30,30])
    writer.add_images("input",imgs,step)
    writer.add_images("output",output,step)
    step+=1

