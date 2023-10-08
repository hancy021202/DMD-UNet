import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import math
#
# # 读取图像
# image = cv2.imread('D:\\CODE\\colorfulDMD\\data\\color\\1_1\\x\\1.tif')
# array = np.array(image)
# # 获取图像的尺寸
# height, width = image.shape[:2]
# # 定义旋转角度（逆时针为正）
# angle = 5
# # 定义缩放因子
# scale = 1.0
# # 计算旋转中心点
# center = (width / 2, height / 2)
# # 使用旋转矩阵进行旋转
# rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
# rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
# # 显示旋转后的图像
# cv2.imshow('Rotated Image', rotated_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite('1_2.tif',rotated_image)

# PIL版本   图片格式不同：PIL打开图像的shape为（w，h），而cv2与plt打开rgb图像都是（h，w，c）。将PIL格式图像转为numpy后，其格式也变为（h，w，c）

# from PIL import Image
# import numpy as np
#
#
# def read_tif_image(file_path):
#     image = Image.open(file_path)
#     # 如果需要将图像转换为RGB格式，可以使用下面的代码
#     image = image.convert("RGB")
#
#     image_array = np.array(image)
#
#     # 如果图像是单通道灰度图，可以将其转换为三通道形式
#     if len(image_array.shape) == 2:
#         image_array = np.expand_dims(image_array, axis=2)
#         image_array = np.repeat(image_array, 3, axis=2)
#
#     # 将通道维度放在第0维
#     image_array = np.transpose(image_array, (2, 0, 1))
#
#     return image_array
#
#
# # 读取.tif图像文件
# image_array = read_tif_image('D:\\CODE\\colorfulDMD\\data\\color\\1_1\\x\\1.tif')
#
# print(image_array.shape)  # 输出图像的形状，例如：(3, 256, 256)
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name())
# print(torch.cuda.get_device_capability(0))
# plt.figure()
# plt.figure()
# plt.show()
# print(torch.__version__)
# print(torch.version.cuda)
image = cv2.imread("G:\\projects\\colorfulPMD\\data\\color\\1_1\\x\\1.tif")
plt.figure()
plt.imshow(image, cmap='gray')  # 原始值
plt.axis('off')
plt.title('image')
plt.show()

image_slice = image[240, :]
plt.figure()
plt.plot(image_slice)  # 误差值
plt.xticks([0, 575])
plt.yticks([0, 80])
plt.grid()
plt.title('image_slice')
plt.show()