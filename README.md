# DMD-UNet
## 工程介绍
    本工程将相位测量偏折术中的相位提取步骤用深度学习网络替代，使之在高精度的同时满足快速计算的需求。
## 数据集处理
    目前可实现黑白数据集以及彩色数据集的训练和预测（论文复现），下一步实现彩色数据集的处理和训练
## UNet修改思路
    下一步拟在UNet中加入注意力机制等一系列来优化网络，例如TransUNet
    ![Image](https://github.com/hancy021202/DMD-UNet/blob/main/picture/UNet.jpg "UNet")