import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import math
# 利用十步相移法求得分子，分母，相位（商）
def load_img(data_path, size):
    signal = os.listdir(data_path)
    count_x = len(signal)  # 数据集的图片总数

    x_up_label = np.empty((count_x, 1, size[0], size[1]), dtype="float32")
    x_down_label = np.empty((count_x, 1, size[0], size[1]), dtype="float32")
    x_img = np.empty((count_x, 1, size[0], size[1]), dtype="float32")
    x_phase_label = np.empty((count_x, 1, size[0], size[1]), dtype="float32")
    num = 0
    for fsingal in signal:
        # 标签(只读取x)
        xm = np.zeros((size[0], size[1]), dtype="float32")
        xn = np.zeros((size[0], size[1]), dtype="float32")
        arr = np.zeros((size[0], size[1]), dtype="float32")
        for i in range(10):
            img = plt.imread(data_path + fsingal + "\\x\\" + "%d" % (i + 1) + ".tif", 0)  # 以灰度图方式读取图片
            arr = np.array(img, dtype="float32")
            xm = xm + arr * math.sin(2 * math.pi * (i + 1) / 10)
            xn = xn + arr * math.cos(2 * math.pi * (i + 1) / 10)
            xp = np.arctan(xm / (xn + 1e-8))
        x_up_label[num, 0, :, :] = xm / 255.0
        x_down_label[num, 0, :, :] = xn / 255.0
        x_img[num, 0, :, :] = arr / 255.0
        x_phase_label[num, 0, :, :] = xp / 255.0
        num = num + 1

    return x_up_label, x_down_label, x_phase_label, x_img


if __name__ == "__main__":
    train_path = 'F:\\projects\\解相unet\\data\\data_new\\train_new\\img_new\\'
    validate_path = 'F:\\projects\\解相unet\\data\\data_new\\val_new\\img_new\\'
    [x_up_train_label, x_down_train_label, x_train_phase, x_train_img] = load_img(train_path, [480, 480])
    [x_up_validate_label, x_down_validate_label, x_validate_phase, x_validate_img] = load_img(validate_path, [480, 480])
    print(x_up_train_label.shape)
    print(x_train_img.shape)
    print(x_down_validate_label.shape)
    print(x_train_phase[20, :, :, :])

###把“0”换到第二位！！！！！！！！！！！！！！！！！！！！！tensorflow:NHWC;    pytorch:NCHW
# indx = 20
# x_img = x_train_img[indx, 0, :, :]
# xm = x_up_train_label[indx, 0, :, :]
# xn = x_down_train_label[indx, 0, :, :]
# x_phase = x_validate_phase[indx, 0, :, :]
# #
# plt.imshow(x_img, cmap='gray')
# plt.axis('off')
# plt.figure()
# #
# plt.imshow(xm, cmap='gray')
# plt.axis('off')
# plt.figure()
# #
# plt.imshow(xn, cmap='gray')
# plt.axis('off')
# plt.figure()
# #
# plt.imshow(np.arctan(xm / xn), cmap='gray')
# plt.axis('off')
# plt.figure()
#
# plt.imshow(x_phase, cmap='gray')
# plt.axis('off')
# plt.figure()
#
# plt.show()
