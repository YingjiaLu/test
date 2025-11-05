from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "data\\train\\bees_image\\92663402_37f379e57a.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

writer.add_image("train",img_array,1,dataformats="HWC")

for i in range(100):
    writer.add_scalar("y=x",i,i)

writer.close()