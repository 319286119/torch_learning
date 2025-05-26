from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer=SummaryWriter("log")

img_path="dataset/train/bees/16838648_415acd9e3f.jpg"
img_PIL=Image.open(img_path)
img_arr=np.array(img_PIL)
writer.add_image("img",img_arr,2,dataformats='HWC')
for i in range(100):
    writer.add_scalar("y=x",i,i)

writer.close()
