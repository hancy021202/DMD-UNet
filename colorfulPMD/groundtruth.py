import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import math


# 利用十步相移法求得分子，分母，相位（商）
def load_label(data_path, size):
    signal = os.listdir(data_path)
    count_x = len(signal)  # 数据集的图片总数

    x_up_label = np.empty((count_x, 1, size[0], size[1]), dtype="float32")
    x_down_label = np.empty((count_x, 1, size[0], size[1]), dtype="float32")
    x_img_gray = np.empty((count_x, 1, size[0], size[1]), dtype="float32")
    x_phase_label = np.empty((count_x, 1, size[0], size[1]), dtype="float32")
    num = 0
    for fsingal in signal:
        # 标签(只读取x)
        xm = np.zeros((size[0], size[1]), dtype="float32")
        xn = np.zeros((size[0], size[1]), dtype="float32")
        arr = np.zeros((size[0], size[1]), dtype="float32")
        for i in range(10):
            img_gray = plt.imread(data_path + fsingal + "\\x\\" + "%d" % (i + 1) + ".tif", 0)  # 以灰度图方式读取图片标签集
            img_gray = img_gray[12:588, 112:688]  # 裁为大小576*576
            arr = np.array(img_gray, dtype="float32")
            xm = xm + arr * math.sin(2 * math.pi * (i + 1) / 10)
            xn = xn + arr * math.cos(2 * math.pi * (i + 1) / 10)
            xp = np.arctan(xm / (xn + 1e-8))
        x_up_label[num, 0, :, :] = xm / 255.0
        x_down_label[num, 0, :, :] = xn / 255.0
        x_img_gray[num, 0, :, :] = arr / 255.0
        x_phase_label[num, 0, :, :] = xp / 255.0
        num = num + 1

    return x_up_label, x_down_label, x_phase_label, x_img_gray


def load_img(data_path, size):
    signal = os.listdir(data_path)
    count_x = len(signal)  # 数据集的图片总数
    x_img_rgb = np.empty((count_x, 3, size[0], size[1]), dtype="float32")
    num = 0
    for fsingal in signal:
        # 标签(只读取x)
        # arr = np.zeros((size[0], size[1]), dtype="float32")
        for i in range(3):
            img_rgb = plt.imread(data_path + fsingal + "\\x\\" + "%d" % (i + 1) + ".tif")  # 以灰度图方式读取图片标签集
            img_rgb = img_rgb[12:588, 112:688]  # 裁为大小576*576
            arr = np.array(img_rgb, dtype="float32")
            x_img_rgb[num, i, :, :] = arr / 255.0
        num = num + 1

    return x_img_rgb


if __name__ == "__main__":
    train_path = 'G:\\projects\\colorfulPMD\\data\\phase_new\\train\\'
    validate_path = 'G:\\projects\\colorfulPMD\\data\\phase_new\\val\\'
    [x_up_train_label, x_down_train_label, x_train_phase, x_train_img] = load_label(train_path, [576, 576])
    [x_up_validate_label, x_down_validate_label, x_validate_phase, x_validate_img] = load_label(validate_path, [576, 576])
    print(x_up_train_label.shape)
    print(x_train_img.shape)
    print(x_down_validate_label.shape)
    print(x_train_phase[5, :, :, :])
    train_path = 'G:\\projects\\colorfulPMD\\data\\color_new\\train\\'
    validate_path = 'G:\\projects\\colorfulPMD\\data\\color_new\\val\\'
    x_train_img = load_img(train_path, [576, 576])
    x_validate_img = load_img(validate_path, [576, 576])
    print(x_validate_img.shape)
    print(x_train_img.shape)
    print(x_train_img[5, :, :, :])
    ##把“0”换到第二位！！！！！！！！！！！！！！！！！！！！！tensorflow:NHWC;    pytorch:NCHW
    indx = 20
    x_img_1 = x_train_img[indx, 0, :, :]
    x_img_2 = x_train_img[indx, 1, :, :]
    x_img_3 = x_train_img[indx, 2, :, :]
    xm = x_up_train_label[indx, 0, :, :]
    xn = x_down_train_label[indx, 0, :, :]
    x_phase_label = x_validate_phase[indx, 0, :, :]

    plt.figure()
    plt.imshow(x_img_1, cmap='gray')
    plt.axis('off')
    plt.title('x_img_1')

    plt.figure()
    plt.imshow(x_img_2, cmap='gray')
    plt.axis('off')
    plt.title('x_img_2')

    plt.figure()
    plt.imshow(x_img_3, cmap='gray')
    plt.axis('off')
    plt.title('x_img_3')

    x_img_1_slice = x_img_1[240, :]
    plt.figure()
    plt.plot(x_img_1_slice)
    plt.xticks([0, 575])
    plt.yticks([0, 1])
    plt.grid()
    plt.title('x_img_l_slice')

    plt.figure()
    plt.imshow(xm, cmap='gray')
    plt.axis('off')
    plt.title('x_up_label_new')

    xm_slice = xm[240, :]
    plt.figure()
    plt.plot(xm_slice)
    plt.xticks([0, 575])
    plt.yticks([0, 1])
    plt.grid()
    plt.title('x_down_label_slice')

    plt.figure()
    plt.imshow(xn, cmap='gray')
    plt.axis('off')
    plt.title('x_phase_label_new')

    xn_slice = xn[240, :]
    plt.figure()
    plt.plot(xn_slice)
    plt.xticks([0, 575])
    plt.yticks([0, 1])
    plt.grid()
    plt.title('x_phase_label_slice')

    x_phase_label_new = np.arctan(xm / xn)
    plt.figure()
    plt.imshow(x_phase_label_new, cmap='gray')
    plt.axis('off')
    plt.title('x_phase_label_new')

    x_phase_label_new_slice = x_phase_label_new[240, :]
    plt.figure()
    plt.plot(x_phase_label_new_slice)
    plt.xticks([0, 575])
    plt.yticks([0, 1])
    plt.grid()
    plt.title('x_phase_label_new_slice')

    plt.figure()
    plt.imshow(x_phase_label, cmap='gray')
    plt.axis('off')
    plt.title('x_phase_label')

    x_phase_label_slice = x_phase_label[240, :]
    plt.figure()
    plt.plot(x_phase_label_slice)
    plt.xticks([0, 575])
    plt.yticks([0, 1])
    plt.grid()
    plt.title('x_phase_label_slice')

    plt.show()
