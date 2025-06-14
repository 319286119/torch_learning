import torch.nn
import torch.nn.functional as F

input=torch.tensor([[1,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]])
input=torch.reshape(input,(1,1,5,5))
kernal=torch.tensor([[1,2,1],
                    [0,1,0],
                     [2,1,0]])
kernal=torch.reshape(kernal,(1,1,3,3))
#print(input.shape)

output=F.conv2d(input,kernal,stride=1)
print(output)

