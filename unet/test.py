import glob
import numpy as np
import torch
import os
import cv2
from PIL import Image
from Unet import *
from torchvision import transforms

weightPath = 'weight/unet.pth'
output_dir = 'result'
output_dir2 = 'output'
input_dir = 'cropped_images'
predict = []
transform = transforms.Compose([
    # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5 ),

    transforms.ToTensor()
])
if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络
    net = UNet()
    # 将网络拷贝到deivce中
    net.to(device)
    # 加载模型参数
    net.load_state_dict(torch.load(weightPath))
    # 测试模式
    net.eval()
    # 读取所有图片路径
    tests_path = sorted(glob.glob('cropped_images/*.png'))

    # 遍历所有图片
    for test_path in tests_path:
        output_filename = f"{test_path}"

        # 保存结果地址
        output_path = os.path.join(output_dir, output_filename)
        output_path1 = os.path.join(output_dir2, output_filename)
        # 读取图片
        img = Image.open(test_path)
        # 转为tensor
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(device)
        # print(img_tensor.shape)
        # 预测
        pred = net(img_tensor)
        pred = pred.squeeze(0)
        pred = pred.transpose(0, 2).transpose(0, 1)
        out_img = pred.cpu().detach().numpy()
        #cv2.imwrite(output_path1, out_img*255)

        # # 处理结果
        out_img[out_img >= 0.9] = 255
        out_img[out_img < 0.9] = 0
        # # 保存图片
        cv2.imwrite(output_path, out_img)

        # cv2.imshow('out', out_img * 255)
        # cv2.waitKey(0)


def getVal():
    return predict
