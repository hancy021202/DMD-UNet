import os
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch import optim
from torch.backends import cudnn
from torch.utils.data import Dataset
from groundtruth import load_label, load_img
from dataset import train_data_Loader
from loss import MS_SSIM_L1_LOSS, au_and_bem_torch, ssim
from src.depthwise_Unet import IUnet
from src.transunet import TransUNet
#
torch.backends.cudnn.enabled = False
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

lr = 0.0001

epochs = 90  # 学习100次
ckpt = 1  # 每学习50次保存一次数据
batch_size = 2  # 每次训练32个数据
model1_path = "Weights_up_gray100.pth"  # 存储路径
#model2_path = "Weights_down100.pth"  # 存储路径
#model3_path = "Weights_phase100.pth"  # 存储路径
################################################################
# ---------------------------warm up---------------------------#
################################################################

# ============= 3) Load Model + Loss + Optimizer + Learn_rate_update ==========#
model1 = IUnet().cuda()#分子计算
if os.path.isfile(model1_path):
    model1.load_state_dict(torch.load(model1_path))  ## Load the pretrained Encoder
    print('IUnet is Successfully Loaded from %s' % model1_path)

# model2 = IUnet().cuda()#分母计算
# if os.path.isfile(model1_path):
#     model2.load_state_dict(torch.load(model2_path))  ## Load the pretrained Encoder
#     print('IUnet is Successfully Loaded from %s' % model2_path)

# model3 = IUnet().cuda()#相位计算
# if os.path.isfile(model1_path):
#     model3.load_state_dict(torch.load(model3_path))  ## Load the pretrained Encoder
#     print('IUnet is Successfully Loaded from %s' % model3_path)

criterion1 = nn.MSELoss(reduction='mean')  ## Define the Loss function MSELoss
criterion2 = nn.L1Loss(size_average=True).cuda()  ## Define the Loss function L1Loss
Ms_ssim = MS_SSIM_L1_LOSS().cuda()
optimizer1 = optim.Adam(model1.parameters(), lr=lr, weight_decay=1e-8, betas=(0.9, 0.999))  ## optimizer 1: Adam
#optimizer2 = optim.Adam(model2.parameters(), lr=lr, weight_decay=1e-8, betas=(0.9, 0.999))  ## optimizer 1: Adam
#optimizer3 = optim.Adam(model3.parameters(), lr=lr, weight_decay=1e-8, betas=(0.9, 0.999))  ## optimizer 1: Adam
# lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.1)   # learning-rate update

# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-7)  ## optimizer 2: SGD
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=180, gamma=0.1)  # learning-rate update: lr = lr* 1/gamma for each step_size = 180

# ============= 4) Tensorboard_show + Save_model ==========#
# if os.path.exists('train_logs'):  # for tensorboard: copy dir of train_logs  ## Tensorboard_show: case 1
#   shutil.rmtree('train_logs')  # ---> console (see tensorboard): tensorboard --logdir = dir of train_logs

writer = SummaryWriter(logdir='./runs1/up100')  ## Tensorboard_show: case 2


# def save_checkpoint(model, epoch):  # save model function
#     model_out_path = 'Weights90.pth'
#     torch.save(model.state_dict(), model_out_path)


###################################################################
# ------------------- Main Train (Run second)----------------------
###################################################################
def train(device, training_loader, validate_loader, start_epoch=0):
    print('Start training...')

    for epoch in range(start_epoch, epochs, 1):

        epoch += 1
        epoch_train_loss1, epoch_val_loss1 = [], []
        #epoch_train_loss2, epoch_val_loss2 = [], []
        #epoch_train_loss3, epoch_val_loss3 = [], []
        # ============Epoch Train=============== #
        model1.train()
        #model2.train()
        #model3.train()

        for image1, label1 in training_loader:  # 100  3
        #for image1, image2, image3, label1, label2, label3 in training_loader:  # 100  3
            optimizer1.zero_grad()  # fixed
            #optimizer2.zero_grad()  # fixed
            #optimizer3.zero_grad()  # fixed
            image1 = image1.to(device=device, dtype=torch.float32)
            #image2 = image2.to(device=device, dtype=torch.float32)
            #image3 = image3.to(device=device, dtype=torch.float32)
            label1 = label1.to(device=device, dtype=torch.float32)
            #label2 = label2.to(device=device, dtype=torch.float32)
            #label3 = label3.to(device=device, dtype=torch.float32)
            image1 = model1(image1)  # call model  输出为结果
            #image2 = model2(image2)  # call model  输出为结果
            #image3 = model3(image3)  # call model  输出为结果
            loss1 = criterion1(image1, label1)  # compute loss
            #loss2 = criterion1(image2, label2)  # compute loss
            #loss3 = criterion1(image3, label3)  # compute loss
            SSIM1 = ssim(image1, label1)
            #SSIM2 = ssim(image2, label2)
            #SSIM3 = ssim(image3, label3)
            # BEM = au_and_bem_torch(image, label)
            epoch_train_loss1.append(loss1.item())  # save all losses into a vector for one epoch
            #epoch_train_loss2.append(loss2.item())  # save all losses into a vector for one epoch
            #epoch_train_loss3.append(loss3.item())  # save all losses into a vector for one epoch
            print('epoch:', epoch, 'loss/train1:', loss1.item(), 'ssimL1:', SSIM1)
            #print('epoch:', epoch, 'loss/train2:', loss2.item(), 'ssimL2:', SSIM2)
            #print('epoch:', epoch, 'loss/train3:', loss3.item(), 'ssimL3:', SSIM3)
            writer.add_scalar("Loss_train1", loss1, epoch)
            #writer.add_scalar("Loss_train2", loss2, epoch)
            #writer.add_scalar("Loss_train3", loss3, epoch)
            loss1.backward()  # fixed
            #loss2.backward()  # fixed
            #loss3.backward()  # fixed
            optimizer1.step()  # fixed
            #optimizer2.step()  # fixed
            #optimizer3.step()  # fixed
            for name, layer in model1.named_parameters():
                # writer.add_histogram('torch/'+name + '_grad_weight_decay', layer.grad, epoch*iteration)
                writer.add_histogram('net/' + name + '_data_weight_decay', layer, epoch)
            # for name, layer in model2.named_parameters():
            #     # writer.add_histogram('torch/'+name + '_grad_weight_decay', layer.grad, epoch*iteration)
            #     writer.add_histogram('net/' + name + '_data_weight_decay', layer, epoch)
            # for name, layer in model3.named_parameters():
            #     # writer.add_histogram('torch/'+name + '_grad_weight_decay', layer.grad, epoch*iteration)
            #     writer.add_histogram('net/' + name + '_data_weight_decay', layer, epoch)

        # lr_scheduler.step()  # if update_lr, activate here!

        t_loss1 = np.nanmean(np.array(epoch_train_loss1))  # compute the mean value of all losses, as one epoch loss
        #t_loss2 = np.nanmean(np.array(epoch_train_loss2))  # compute the mean value of all losses, as one epoch loss
        #t_loss3 = np.nanmean(np.array(epoch_train_loss3))  # compute the mean value of all losses, as one epoch loss
        writer.add_scalar('Loss/t_loss1', t_loss1, epoch)  # write to tensorboard to check
        print('Epoch:', epochs, '/', epoch, 'training loss1:', t_loss1)  # print loss for each epoch
        # writer.add_scalar('Loss/t_loss2', t_loss2, epoch)  # write to tensorboard to check
        # print('Epoch:', epochs, '/', epoch, 'training loss2:', t_loss2)  # print loss for each epoch
        # writer.add_scalar('Loss/t_loss3', t_loss3, epoch)  # write to tensorboard to check
        # print('Epoch:', epochs, '/', epoch, 'training loss3:', t_loss3)  # print loss for each epoch

        if epoch % ckpt == 0:  # if each ckpt epochs, then start to save model
            torch.save(model1.state_dict(), 'Weights_up100_gray.pth')
            #torch.save(model2.state_dict(), 'Weights_down1.pth')
            #torch.save(model3.state_dict(), 'Weights_phase90.pth')

        # ============Epoch Validate=============== #
        if epoch % 10 == 0:
            model1.eval()  # fixed
            #model2.eval()  # fixed
            #model3.eval()  # fixed
            with torch.no_grad():  # fixed
                for image1, label1 in validate_loader:
                #for image1, image2, image3, label1, label2, label3 in validate_loader:
                    image1 = image1.to(device=device, dtype=torch.float32)
                    #image2 = image2.to(device=device, dtype=torch.float32)
                    #image3 = image3.to(device=device, dtype=torch.float32)
                    label1 = label1.to(device=device, dtype=torch.float32)
                    #label2 = label2.to(device=device, dtype=torch.float32)
                    #label3 = label3.to(device=device, dtype=torch.float32)
                    image1 = model1(image1)  # call model  输出为结果\
                    #image2 = model2(image2)  # call model  输出为结果
                    #image3 = model3(image3)  # call model  输出为结果
                    loss1 = criterion1(image1, label1)  # compute loss
                    #loss2 = criterion1(image2, label2)  # compute loss
                    #loss3 = criterion1(image3, label3)  # compute loss
                    epoch_val_loss1.append(loss1.item())
                    #epoch_val_loss2.append(loss2.item())
                    #epoch_val_loss3.append(loss3.item())

        if epoch % 10 == 0:
            v_loss1 = np.nanmean(np.array(epoch_val_loss1))
            #v_loss2 = np.nanmean(np.array(epoch_val_loss2))
            #v_loss3 = np.nanmean(np.array(epoch_val_loss3))
            writer.add_scalar('val/v_loss1', v_loss1, epoch)
            print(' validate loss1:', v_loss1)
            # writer.add_scalar('val/v_loss2', v_loss2, epoch)
            # print(' validate loss2:', v_loss2)
            # writer.add_scalar('val/v_loss3', v_loss3, epoch)
            # print(' validate loss3:', v_loss3)

    writer.close()  # close tensorboard


###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == "__main__":
    # 数据集载入
    train_img_path = 'G:\\projects\\colorfulPMD\\data\\color_new\\train\\'
    validate_img_path = 'G:\\projects\\colorfulPMD\\data\\color_new\\val\\'
    #导入彩色图
    x_train_img_rgb = load_img(train_img_path, [576, 576])
    x_validate_img_rgb = load_img(validate_img_path, [576, 576])
    # 标签载入
    train_label_path = 'G:\\projects\\colorfulPMD\\data\\phase_new\\train\\'
    validate_label_path = 'G:\\projects\\colorfulPMD\\data\\phase_new\\val\\'
    #计算真值
    [x_up_train_label, x_down_train_label, x_phase_train_label, x_train_img_gray] = load_label(train_label_path, [576, 576])
    [x_up_validate_label, x_down_validate_label, x_phase_validate_label, x_validate_img_gray] = load_label(validate_label_path, [576, 576])
    #创建dataset
    train_dataset = train_data_Loader(x_train_img_rgb, x_down_train_label) #, x_down_train_label, x_phase_train_label)
    validate_dataset = train_data_Loader(x_validate_img_rgb, x_down_validate_label) #, x_down_validate_label, x_phase_validate_label)
    #载入数据集
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset, batch_size=batch_size, shuffle=False)
    # GPU选择
    torch.cuda.empty_cache()  # 清空内存，防止out of memory
    print("GPU:", torch.cuda.is_available())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train(device, train_loader, validate_loader)
