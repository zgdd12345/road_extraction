import os


# 是否为图片
def is_img(file):
    img_type = os.path.splitext(file)[1].lower()  # 获取扩展名
    if img_type in ['.jpg', '.png', '.jpeg', '.bmp']:
        return True
    else:
        return False
