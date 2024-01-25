import os

import cv2
import numpy as np
import torch
from Read import *
from torch.utils.data import Dataset
from torchvision import transforms

transform = transforms.Compose([
    # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5 ),

    transforms.ToTensor()
])


class Data(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'labels'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        labelName = self.name[index]  # 数据集中标签为png,训练集为jpg
        labelPath = os.path.join(self.path, 'labels', labelName)
        imagePath = os.path.join(self.path, 'inputs', labelName.replace('png', 'jpg'))
        label = read(labelPath)
        image = read(imagePath)

        return transform(image), transform(label)
        # return image,label


if __name__ == '__main__':
    data = Data('train2')
    print(data[0][0].shape)
    print(data[0][1].shape)
    a1 = data[0][0]
    a2 = data[0][1]
    print()
    # from torchvision.transforms import ToPILImage

    # show = ToPILImage()  # 可以把Tensor转成Image，方便可视化
    # show(data[0][0]).show()
    # show(data[0][1]).show()
