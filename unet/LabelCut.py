import os
from PIL import Image
from Read import *
import torchvision.transforms.functional as TF

'''
进行图片裁剪，将所需要的图片裁剪到256*256的分辨率，以适应u-net网络
'''
# 路径和参数设置
input_dir = "groundtruth"  # 输入图像所在的文件夹路径
output_dir = "cropped_images_label"  # 裁剪后的图像保存路径
crop_size = 256  # 裁剪后的大小

# 创建保存目录
os.makedirs(output_dir, exist_ok=True)

# 遍历文件夹中的图像文件
image_files = os.listdir(input_dir)
for k,image_file in enumerate(image_files):
    image_path = os.path.join(input_dir, image_file)
    if os.path.isfile(image_path):
        # 打开图像
        image = Image.open(image_path)

        # 确定裁剪的行数和列数
        width, height = image.size
        num_rows = height // crop_size+1
        num_cols = width // crop_size+1

        # 循环裁剪图像并保存
        for i in range(num_rows):
            for j in range(num_cols):
                left = j * crop_size
                top = i * crop_size
                right = left + crop_size
                bottom = top + crop_size

                crop = TF.crop(image, top, left, crop_size, crop_size)
                output_filename = f"cropped_{k}_{i}_{j}_.png"
                output_path = os.path.join(output_dir, output_filename)
                crop.save(output_path)
