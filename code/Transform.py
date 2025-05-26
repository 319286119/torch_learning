from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path="dataset/train/ants/0013035.jpg"
img=Image.open(img_path)

#transform(数据类型的转化)的使用

#totensor的使用
tensor_trans=transforms.ToTensor()#transform相当于是工具箱，totensor相当于是具体的工具
tensor_img=tensor_trans(img)
writer=SummaryWriter("log")
writer.add_image("img",tensor_img,3)

#normalize的使用
#第一个维度时三个信道（三种颜色）,第二个维度是高度，第三个四宽度
trans_norm=transforms.Normalize([1000,3,2],[100,0.5,0.5])#标准化数据的，加快模型训练的效率
norm_image=trans_norm(tensor_img)
print(norm_image[0][0][0])
writer.add_image("img",tensor_img,2)


#resize
#重塑图片大小
trans_resize=transforms.Resize(512)
img_resize=trans_resize(img)

#compose
#对变化操进行打包
trans_compose=transforms.Compose([
    trans_resize,
    tensor_trans,
])
compose_img=trans_compose(img)

#randomcrop
#随机截取
trans_rand=transforms.RandomCrop(512)
for i in range(10):
    img_rand = trans_rand(img)
    img_rand_ter=tensor_trans(img_rand)
    writer.add_image("rand",img_rand_ter,i)

writer.close()
