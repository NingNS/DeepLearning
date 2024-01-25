import os
from PIL import Image

# 路径和参数设置
input_dir = "result"  # 裁剪后的图像所在的文件夹路径
output_dir = "final_out_image"  # 重新拼接后的图像保存路径
crop_size = 256  # 裁剪后的大小
count = 0
# 创建保存目录
os.makedirs(output_dir, exist_ok=True)

# 遍历文件夹中的图像文件
image_files = os.listdir(input_dir)
image_files.sort()  # 按文件名排序，确保拼接的顺序

# 计算拼接后图像的大小
num_rows = 0
num_cols = 0
for image_file in image_files:
    if image_file.startswith("cropped_"):
        num_rows = max(num_rows, int(image_file.split("_")[2]) + 1)
        num_cols = max(num_cols, int(image_file.split("_")[3]) + 1)

# 创建拼接后的图像
reconstructed_image = Image.new("RGB", (num_cols * crop_size, num_rows * crop_size))

# 拼接图像
for image_file in image_files:
    if image_file.startswith("cropped_"):
        i, j, _ = image_file.split("_")[2:]
        i = int(i)
        j = int(j)

        image_path = os.path.join(input_dir, image_file)
        try:
            image = Image.open(image_path)
        except:
            continue  # 跳过无法打开的图像

        left = j * crop_size
        top = i * crop_size
        right = left + crop_size
        bottom = top + crop_size

        reconstructed_image.paste(image, (left, top, right, bottom))
        count += 1

    if count == num_rows*num_cols:
        count = 0
        # 检查拼接后的图像是否为空
        if reconstructed_image.size[0] == 0 or reconstructed_image.size[1] == 0:
            print("拼接后的图像为空，无法保存。请检查裁剪图像是否存在。")
        else:
            # 保存拼接后的图像
            output_filename = f"{image_file}"
            output_path = os.path.join(output_dir, output_filename)
            reconstructed_image.save(output_path)
