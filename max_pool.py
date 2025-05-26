import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("CIFAR10",train=False,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset,batch_size=64)

class Max_pool(nn.Module):
    def __init__(self):
        super(Max_pool,self).__init__()
        self.maxpool=nn.MaxPool2d(kernel_size=3,ceil_mode=False)
    def forward(self,x):
        return self.maxpool(x)

max_pool=Max_pool()
step=1
writer=SummaryWriter("log")
for data in dataloader:
    img,target=data
    writer.add_images(tag="input",img_tensor=img,global_step=step)
    output=max_pool(img)
    writer.add_images(tag="output",img_tensor=output,global_step=step)
    step+=1
