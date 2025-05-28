import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Sequential import My_Layers

train_datasets=torchvision.datasets.CIFAR10(root="CIFAR10",
                                            train=True,
                                            download=True,
                                            transform=torchvision.transforms.ToTensor())
test_datasets=torchvision.datasets.CIFAR10(root="CIFAR10",
                                           train=False,
                                           download=True,
                                           transform=torchvision.transforms.ToTensor())

#每次取64个的
train_loader=DataLoader(dataset=train_datasets,batch_size=64)
test_loader=DataLoader(dataset=test_datasets,batch_size=64)

model=My_Layers()
if torch.cuda.is_available():
    model=model.cuda()

loss_func=nn.CrossEntropyLoss()#损失函数
loss_func=loss_func.cuda()

optimizer=torch.optim.SGD(model.parameters(),lr=1e-2)#优化器

test_step=1
train_step=1
epoch=10
writer=SummaryWriter(log_dir="log")
for i in range(epoch):
    total_test_loss = 0
    total_train_loss = 0
    print("第{}轮开始".format(i+1))
    #开始训练
    #切换为训练模式
    model.train()#只对少部分的模型适用
    for data in train_loader:
        img,target=data
        if torch.cuda.is_available():
            img=img.cuda()
            target=target.cuda()
        output = model(img)
        loss = loss_func(output, target)
        total_train_loss+=loss

        #优化的统一格式
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #end

        train_step+=1
        if train_step%100==0:
            print("训练次数{},loss={}".format(train_step,loss))
            writer.add_scalar(tag="train_loss",scalar_value=total_train_loss,global_step=train_step)

    total_accuracy=0
    #开始测试
    model.eval()#切换为
    with torch.no_grad():#没有梯度变化
        for data in test_loader:
            img,target=data
            if torch.cuda.is_available():
                img=img.cuda()
                target=target.cuda()
            output = model(img)
            loss=loss_func(output,target)
            total_test_loss+=loss
            accuracy=(output.argmax(1)==target).sum()#将每个图像对比后相加
            total_accuracy+=accuracy
    print("第{}轮测试集上的loss:{}".format(i,total_test_loss))
    print("整体测试集上的正确率{}".format(total_accuracy/len(test_datasets)))
    writer.add_scalar(tag="test_accuracy",scalar_value=total_accuracy,global_step=test_step)
    test_step+=1

    torch.save(model,"model{}".format(i))

