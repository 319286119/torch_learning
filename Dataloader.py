import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10(root='CIFAR10', train=False,transform=torchvision.transforms.ToTensor())
test_loader=DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=False)
#第一个参数设置数据集，第二个设置每次读取的张数，第三个随机取，第四个线程数，第五个是否舍弃不足一次数量的、剩下的图片

writer=SummaryWriter("log")
step=0
for data in test_loader:
    imgs,targets=data
    #print(imgs.shape)
    #print(targets)

    writer.add_images("imgs",imgs,step)
    step+=1

writer.close()

