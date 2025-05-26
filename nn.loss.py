import torch
from torch import nn

inputs=torch.tensor([1,2,3],dtype=torch.float32)
target=torch.tensor([1,2,5],dtype=torch.float32)

inputs=torch.reshape(inputs,[1,1,1,3])
target=torch.reshape(target,[1,1,1,3])

loss=nn.L1Loss(reduction="sum")
print(loss(inputs,target))

loss_mse=nn.MSELoss(reduction="mean")
print(loss_mse(inputs,target))

x=torch.tensor([1,2,3],dtype=torch.float32)
x=x.reshape((1,3))#一个样本在三个种类的不同概率
y=torch.tensor([1])#每个样本真实的类别
loss_cross=nn.CrossEntropyLoss()
print(loss_cross(x,y))