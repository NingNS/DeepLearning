from PIL import Image


# 等比例处理图片
def resize_image(path, size=(636, 424)):
    img = Image.open(path)
    maxLength = max(img.size)
    mask = Image.new('RGB', (maxLength, maxLength), (0, 0, 0))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask


# 直接读图片
def read(path):
    img = Image.open(path)
    return img
