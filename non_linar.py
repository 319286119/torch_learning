import torchvision
from torch import nn
from torch.nn import Sigmoid
from torch.utils.data import DataLoader

from max_pool import writer

dataset=torchvision.datasets.CIFAR10(root="CIFAR10",train=False,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64)

class No_linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.no_linear=Sigmoid()
    def forward(self,x):
        return self.no_linear(x)

step=1
nolinear=No_linear()
for data in dataloader:
    img,target=data
    writer.add_images(tag="input",img_tensor=img,global_step=step)
    output=nolinear(img)
    writer.add_images(tag="output",img_tensor=output,global_step=step)
    step+=1

