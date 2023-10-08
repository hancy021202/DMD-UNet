# from src.unet_model import UNet
import numpy as np

from src.RepUnet import Unwrapping
# from src.depthwise_Unet_LeakyRelu import IUnet
from src.depthwise_Unet import IUnet
from dataset import data_Loader
from torch import optim
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch
import groundtruth
import torchvision

def train_net(net, device, data_path, epochs=100, batch_size=1, lr=0.0001):
    # 加载训练集
    train_path = 'D:\\活动\\大创\\工程文件\\解相unet\\data\\'  # data_img\\new20000_1_1\\x
    [x_up_label, x_down_label,x_img] = groundtruth.load_truth(train_path, [480, 480])
    train_dataset = data_Loader(x_img, x_up_label)
    test_dataset = data_Loader(x_img, x_up_label)
    print("数据个数：", len)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    # 定义Adam算法
    optimizer = optim.Adam(net.parameters(), lr=lr, eps=1e-8, betas=(0.9,0.999))
    # 定义Loss算法

    criterion = nn.MSELoss(reduction='mean')
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        print('epoch:', epoch+1)
        for image, label in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # print("image_size:",image.size(),"  label_size",label.size())
            # 使用网络参数，输出预测结果
            pred = net(image)
            # pred_array=np.array(pred)
            # writer.add_image("up_pre", pred_array,100)#tensorflow:NHWC;    pytorch:NCHW(默认)
            # 计算loss
            loss = criterion(pred, label)
            print('epoch:',epoch+1,'  Loss_train:', loss.item())
            writer.add_scalar("Loss_train", loss, epoch + 1)
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model_depthwise_unet1up.pth.pth')
            # 更新参数
                writer.add_scalar("Loss_train", best_loss, epoch + 1)

                loss.backward()
                optimizer.step()

def save_checkpoint(model, epoch):  # save model function
    model_out_path = 'Weights' + '/' + "{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)



###################################################################
# ------------------- Main Train (Run second)----------------------
###################################################################
def train(training_data_loader, validate_data_loader, start_epoch=0):
    print('Start training...')

    for epoch in range(start_epoch, epochs, 1):

        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []

        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(training_data_loader):  # 100  3

            img, gt = batch['img'], batch['label']
            img = img.type(torch.FloatTensor).cuda()
            gt = gt.type(torch.FloatTensor).cuda()
            optimizer.zero_grad()  # fixed
            sr = model(img)  # call model  输出为结果
            loss = Ms_ssim(sr, gt)  # compute loss
            SSIM = ssim(sr,gt)
            BEM = au_and_bem_torch(sr,gt ,False)
            epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch
            print('Epoch: {} Batches:{} training loss: {:.7f} 相似度:{:.4f} BEM:{:.4f}'.format(epoch, iteration, loss, SSIM ,BEM[0]))
            loss.backward()  # fixed
            optimizer.step()  # fixed
            for name, layer in model.named_parameters():
                # writer.add_histogram('torch/'+name + '_grad_weight_decay', layer.grad, epoch*iteration)
                writer.add_histogram('net/' + name + '_data_weight_decay', layer, epoch * iteration)

        #lr_scheduler.step()  # if update_lr, activate here!

        t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
        writer.add_scalar('Loss/t_loss', t_loss, epoch)  # write to tensorboard to check
        print('Epoch: {}/{} training loss: {:.7f}'.format(epochs, epoch, t_loss))  # print loss for each epoch

        if epoch % ckpt == 0:  # if each ckpt epochs, then start to save model
            save_checkpoint(model, epoch)

        # ============Epoch Validate=============== #
        if epoch % 10 == 0:
            model.eval()  # fixed
            with torch.no_grad():  # fixed
                for iteration, batch in enumerate(validate_data_loader, 1):
                    img, gt = batch['img'], batch['label']
                    img = img.type(torch.FloatTensor).cuda()
                    gt = gt.type(torch.FloatTensor).cuda()
                    sr = model(img)
                    loss = Ms_ssim(sr, gt)
                    epoch_val_loss.append(loss.item())

        if epoch % 10 == 0:
            v_loss = np.nanmean(np.array(epoch_val_loss))
            writer.add_scalar('val/v_loss', v_loss, epoch)
            print('             validate loss: {:.7f}'.format(v_loss))

    writer.close()  # close tensorboard


if __name__ == "__main__":
    # 激活tensorboard
    writer = SummaryWriter(log_dir='runs100up')
    # 选择设备，有cuda用cuda，没有就用cpu
    torch.cuda.empty_cache() # 清空内存，防止out of memory
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道1，分类为1
    net = Unwrapping()
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = "data/"
    train_net(net, device, data_path)
    writer.flush()
