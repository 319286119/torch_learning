import torch.nn
import torch

class My_nn(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        output=input+1
        return output


nn=My_nn()
input=torch.tensor(1.0)
output=nn(input)
print(output)
