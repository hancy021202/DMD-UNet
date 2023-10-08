import os
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.backends import cudnn
import groundtruth
from torch.utils.data import Dataset

from dataset import data_Loader
from loss import MS_SSIM_L1_LOSS
from src.depthwise_Unet import IUnet

# from src.RepUnet import Unwrapping

###################################################################
# ------------------- Pre-Define Part----------------------
###################################################################L
# ============= 1) Pre-Define =================== #
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# cudnn.benchmark = True  ###自动寻找最优算法
cudnn.deterministic = True
cudnn.benchmark = False
# ============= 2) HYPER PARAMS(Pre-Defined) ==========#

batch_size = 1  # 每次训练32个数据
# ============= 3) Load Model + Loss + Optimizer + Learn_rate_update ==========#
model1_path = "Weights_up400.pth"  # 存储路径
model2_path = "Weights_down400.pth"  # 存储路径
model3_path = "Weights_phase400.pth"  # 存储路径
model1 = IUnet().cuda()
if os.path.isfile(model1_path):
    model1.load_state_dict(torch.load(model1_path))  ## Load the pretrained Encoder
    print('IUnet is Successfully Loaded from %s' % model1_path)

model2 = IUnet().cuda()
if os.path.isfile(model1_path):
    model2.load_state_dict(torch.load(model2_path))  ## Load the pretrained Encoder
    print('IUnet is Successfully Loaded from %s' % model2_path)

model3 = IUnet().cuda()
if os.path.isfile(model1_path):
    model3.load_state_dict(torch.load(model3_path))  ## Load the pretrained Encoder
    print('IUnet is Successfully Loaded from %s' % model3_path)

criterion1 = nn.MSELoss(reduction='mean')  ## Define the Loss function MSELoss
criterion2 = nn.L1Loss(size_average=True).cuda()  ## Define the Loss function L1Loss
Ms_ssim = MS_SSIM_L1_LOSS().cuda()


def predict(device, validate_loader):
    print('Start predicting...')
    # ============Predict=============== #
    model1.load_state_dict(torch.load('Weights_up90.pth', map_location=device))
    model2.load_state_dict(torch.load('Weights_down90.pth', map_location=device))
    model3.load_state_dict(torch.load('Weights_phase90.pth', map_location=device))
    model1.eval()  # fixed
    model2.eval()  # fixed
    model3.eval()  # fixed
    with torch.no_grad():  # fixed
        for image1, image2, image3, label1, label2, label3 in validate_loader:
            image1 = image1.to(device=device, dtype=torch.float32)
            image2 = image2.to(device=device, dtype=torch.float32)
            image3 = image3.to(device=device, dtype=torch.float32)
            label1 = label1.to(device=device, dtype=torch.float32)
            label2 = label2.to(device=device, dtype=torch.float32)
            label3 = label3.to(device=device, dtype=torch.float32)
            image1 = model1(image1)  # call model  输出为结果\
            image2 = model2(image2)  # call model  输出为结果
            image3 = model3(image3)  # call model  输出为结果
            loss1 = criterion1(image1, label1)  # compute loss
            loss2 = criterion1(image2, label2)  # compute loss
            loss3 = criterion1(image3, label3)  # compute loss
            print('loss:', loss1.item())
            print('loss:', loss2.item())
            print('loss:', loss3.item())
            image1 = np.array(image1.data.cpu()[0])[0]
            image2 = np.array(image2.data.cpu()[0])[0]
            image3 = np.array(image3.data.cpu()[0])[0]
            image4 = np.arctan(image1 / (image2 + 1e-8))
            label1 = np.array(label1.data.cpu()[0])[0]
            label2 = np.array(label2.data.cpu()[0])[0]
            label3 = np.array(label3.data.cpu()[0])[0]
            image1 = image1 / 1.0
            image2 = image2 / 1.0
            image3 = image3 / 1.0
            image4 = image4 / 255.0
            label1 = label1 / 1.0
            label2 = label2 / 1.0
            label3 = label3 / 1.0
        return image1, image2, image3, image4, label1, label2, label3


if __name__ == "__main__":

    # 数据集载入
    validate_path = 'G:\\projects\\解相unet\\data\\data_new\\val_new\\img_new\\'
    [x_up_validate_label, x_down_validate_label, x_phase_validate_label, x_validate_img] = groundtruth.load_img(
        validate_path, [480, 480])
    validate_dataset = data_Loader(x_validate_img, x_up_validate_label, x_down_validate_label, x_phase_validate_label)
    # validate_dataset = data_Loader(x_validate_img, x_down_validate_label)
    validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset, batch_size=batch_size, shuffle=False)
    # GPU选择
    # 选择设备，有cuda用cuda，没有就用cpu
    print("GPU:", torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    [x_up_pred, x_down_pred, x_phase_pred, x_phase_pred_new, x_up_label, x_down_label, x_phase_label] = predict(device,validate_loader)
    ######################################################预测分子
    plt.figure()
    plt.imshow(x_up_pred, cmap='gray')  # 预测值
    plt.axis('off')
    plt.title('x_up_pred')

    plt.figure()
    plt.imshow(x_up_label, cmap='gray')  # 原始值
    plt.axis('off')
    plt.title('x_up_label')

    error_x_up = abs(x_up_pred - x_up_label)

    plt.figure()
    plt.imshow(error_x_up, cmap='gray')  # 误差值
    plt.axis('off')
    plt.title('error_x_up')
    plt.show()

    sum_error1 = 0
    for i in range(480):
        for j in range(480):
            sum_error1 = sum_error1 + error_x_up[i, j]
    mae_up = sum_error1 / 480 / 480
    print('mae_up:', mae_up)
    #######################################################预测分母
    plt.figure()
    plt.imshow(x_down_pred, cmap='gray')  # 预测值
    plt.axis('off')
    plt.title('x_down_pred')

    plt.figure()
    plt.imshow(x_down_label, cmap='gray')  # 原始值
    plt.axis('off')
    plt.title('x_down_label')

    error_x_down = abs(x_down_pred - x_down_label)

    plt.figure()
    plt.imshow(error_x_down, cmap='gray')  # 误差值
    plt.axis('off')
    plt.title('error_x_down')
    plt.show()

    sum_error2 = 0
    for i in range(480):
        for j in range(480):
            sum_error2 = sum_error2 + error_x_down[i, j]
    mae_down = sum_error2 / 480 / 480
    print('mae_down:', mae_down)
    #######################################################直接预测相位
    plt.figure()
    plt.imshow(x_phase_pred, cmap='gray')  # 预测值
    plt.axis('off')
    plt.title('x_phase_pred')

    plt.figure()
    plt.imshow(x_phase_label, cmap='gray')  # 原始值
    plt.axis('off')
    plt.title('x_phase_label')

    error_x_phase = abs(x_phase_pred - x_phase_label)
    error_x_phase_slice = error_x_phase[240, :]

    plt.figure()
    plt.imshow(error_x_phase, cmap='gray')  # 误差值
    plt.axis('off')
    plt.title('error_x_phase')

    plt.figure()
    plt.plot(error_x_phase_slice)  # 误差值
    plt.xticks([0, 479])
    plt.yticks([0, 0.00006])
    plt.grid()
    plt.title('error_x_phase_slice')
    plt.show()

    sum_error3 = 0
    for i in range(480):
        for j in range(480):
            sum_error3 = sum_error3 + error_x_phase[i, j]
    mae_phase = sum_error3 / 480 / 480
    print('mae_phase:', mae_phase)
    #######################################################求商预测相位
    # x_phase_pred_new = np.arctan(x_up_label / (x_down_label+1e-8))
    # x_phase_pred_new = x_phase_pred_new / 255.0

    plt.figure()
    plt.imshow(x_phase_pred_new, cmap='gray')
    plt.axis('off')
    plt.title('x_phase_pred_new')

    plt.figure()
    plt.imshow(x_phase_label, cmap='gray')  # 原始值
    plt.axis('off')
    plt.title('x_phase_label')

    error_x_phase_new = abs(x_phase_pred_new - x_phase_label)
    error_x_phase_new_slice = error_x_phase_new[240, :]

    plt.figure()
    plt.imshow(error_x_phase_new, cmap='gray')  # 误差值
    plt.axis('off')
    plt.title('error_x_phase_new')

    plt.figure()
    plt.plot(error_x_phase_new_slice)  # 误差值
    plt.xticks([0, 479])
    plt.yticks([0, 0.006])
    plt.grid()
    plt.title('error_x_phase_new_slice')
    plt.show()

    sum_error4 = 0
    for i in range(480):
        for j in range(480):
            sum_error4 = sum_error4 + error_x_phase_new[i, j]
    mae_phase_new = sum_error4 / 480 / 480
    print('mae_phase_new:', mae_phase_new)
