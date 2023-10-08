import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np



class MS_SSIM_L1_LOSS(nn.Module):
    """
    Have to use cuda, otherwise the speed is too slow.
    Both the group and shape of input image should be attention on.
    I set 255 and 1 for gray image as default.
    """

    def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
                 data_range=255.0,
                 K=(0.01, 0.03),  # c1,c2
                 alpha=0.025,  # weight of ssim and l1 loss
                 compensation=200.0,  # final factor for total loss
                 cuda_dev=0,  # cuda device choice
                 channel=1):  # RGB image should set to 3 and Gray image should be set to 1
        super(MS_SSIM_L1_LOSS, self).__init__()
        self.channel = channel
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation = compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros(
            (self.channel * len(gaussian_sigmas), 1, filter_size, filter_size))  # 创建了(3*5, 1, 33, 33)个masks
        for idx, sigma in enumerate(gaussian_sigmas):
            if self.channel == 1:
                # only gray layer
                g_masks[idx, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            elif self.channel == 3:
                # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
                g_masks[self.channel * idx + 0, 0, :, :] = self._fspecial_gauss_2d(filter_size,
                                                                                   sigma)  # 每层mask对应不同的sigma
                g_masks[self.channel * idx + 1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
                g_masks[self.channel * idx + 2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            else:
                raise ValueError
        self.g_masks = g_masks.cuda(cuda_dev)  # 转换为cuda数据类型

    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution

        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)
        # Outer product of input and vec2. If input is a vector of size nn and vec2 is a vector of size mm,
        # then out must be a matrix of size (n \times m)(n×m).

    def forward(self, x, y):
        b, c, h, w = x.shape
        assert c == self.channel

        mux = F.conv2d(x, self.g_masks, groups=c, padding=self.pad)  # 图像为96*96，和33*33卷积，出来的是64*64，加上pad=16,出来的是96*96
        muy = F.conv2d(y, self.g_masks, groups=c, padding=self.pad)  # groups 是分组卷积，为了加快卷积的速度

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=c, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=c, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=c, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l = (2 * muxy + self.C1) / (mux2 + muy2 + self.C1)  # [B, 15, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)
        if self.channel == 3:
            lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]  # 亮度对比因子
            PIcs = cs.prod(dim=1)
        elif self.channel == 1:
            lM = l[:, -1, :, :]
            PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM * PIcs  # [B, H, W]

        loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, C, H, W]
        # average l1 loss in num channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-self.channel, length=self.channel),
                               groups=c, padding=self.pad).mean(1)  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation * loss_mix

        return loss_mix.mean()


# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=98):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 98
        else:
            max_val = 98

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def au_and_bem_torch(nn_output, ground_truth):
    """
    difference from "au_and_bem' is converting to np.ndarray and abs()

    calculates Binary Error Map (BEM) and Accuracy of Unwrapping (AU)
    for batch [batch_images,0,width,heidth] and returns mean AU of a batch
    with list of AU for every image and may be with BEM (optionally)

    function returns:
    au_mean - float, mean AU for batch
    au_list - list, info about AU for every image in batch
    bem - 3d boolean tensor, shows BEM in format [images_in_batch,width,height]

    args:
    nn_output - ndarray or torch.tensor - tensor that goes forward the net
    ground_truth - ndarray or tensor - ground truth image (original phase)
    calc_bem - boolean, if needed, will calculate BEM


    with input as np.ndarray runs 10 times faster
    """
    #min_height = torch.min(nn_output)
    nn_output = nn_output.cpu().detach().numpy()
    ground_truth = ground_truth.cpu().detach().numpy()
    bem = np.empty([
        len(nn_output[:, 0, 0, 0]),
        len(nn_output[0, 0, :, 0]),
        len(nn_output[0, 0, 0, :])
    ])
    cnt = 0.
    for k in range(len(nn_output[:, 0, 0, 0])):
        min_height = np.min(ground_truth[k,0,:,:])
        for i in range(len(nn_output[0, 0, :, 0])):
            for j in range(len(nn_output[0, 0, 0, :])):
                x = abs(nn_output[k, 0, i, j] - ground_truth[k, 0, i, j])
                if x <= (ground_truth[k, 0, i, j] - min_height) * 0.05:
                    cnt += 1
    cnt = cnt/(256*256*len(nn_output[:, 0, 0, 0]))
    return cnt


def bem_au(image,gt):
    b,c,h,w = gt.size()
    image = image.detach()
    gt = gt.detach()
    x = torch.min(gt.flatten(2),dim=-1)[0].unsqueeze(1).unsqueeze(1)
    x = (gt-x)*0.05
    image = torch.abs_(gt - image)
    x = torch.le(image, x)
    x = torch.sum(x).item()
    x /= (256*256*b)
    return  x

def CEloss(image,pred,gt,device):
    b,c,h,w = image.size()
    #image = image.flatten(1)
    #pred = pred.flatten(1)
    #gt = gt.flatten(1)
    pred = torch.div(pred-image,3.14,rounding_mode='trunc')
    gt =torch.div(gt-image,3.14,rounding_mode='trunc')
    loss_function = nn.KLDivLoss(reduction='mean').to(device)
    loss = -loss_function(pred,gt)/(256*256*b)
    return loss
