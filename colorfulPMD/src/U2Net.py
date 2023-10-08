import torch
import torch.nn as nn


class BasicLayer(nn.Module):
    def __init__(self, in_dim, out_dim, down=None, up=None):
        super(BasicLayer, self).__init__()
        self.down = down
        self.up = up
        self.pad = nn.ReplicationPad2d(padding=(1, 1, 1, 1))
        if self.down is not None:
            self.down = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, dilation=1, ceil_mode=False)
        if self.up is not None:
            # self.up = nn.ConvTranspose2d(in_channels= out_dim, out_channels= out_dim,kernel_size=4,stride=12,padding=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.block = nn.Sequential(nn.ReplicationPad2d(padding=(1, 1, 1, 1)),
                                   nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=1,
                                             padding=0, bias=False),
                                   nn.BatchNorm2d(out_dim),
                                   nn.LeakyReLU(inplace=True))

    def forward(self, x):
        if self.down is not None:
            x = self.down(x)
        x = self.block(x)
        if self.up is not None:
            x = self.up(x)
        return x


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)



class U_Netblock_7(nn.Module):
    def __init__(self, in_dim, out_dim, depths=14):
        super(U_Netblock_7, self).__init__()
        self.num = depths
        self.block = nn.ModuleList()
        for i in range(self.num):
            if i == 0:
                blk = BasicLayer(in_dim=in_dim, out_dim=out_dim)
            else:
                blk = BasicLayer(in_dim=out_dim if 1 <= i <= 7 else 2 * out_dim, out_dim=out_dim,
                                 down=True if 2 <= i <= 7 else None, up=True if 7 <= i <= 12 else None)
            self.block.append(blk)

    def forward(self, x):
        i = 0
        for layer in self.block:
            if i == 0:
                x = layer(x)
            elif i == 1:
                x1 = layer(x)
            elif i == 2:
                x2 = layer(x1)
            elif i == 3:
                x3 = layer(x2)
            elif i == 4:
                x4 = layer(x3)
            elif i == 5:
                x5 = layer(x4)
            elif i == 6:
                x6 = layer(x5)
            elif i == 7:
                x7 = layer(x6)
                x7 = torch.cat([x6, x7], dim=1)
            elif i == 8:
                x7 = layer(x7)
                x7 = torch.cat([x5, x7], dim=1)
            elif i == 9:
                x7 = layer(x7)
                x7 = torch.cat([x4, x7], dim=1)
            elif i == 10:
                x7 = layer(x7)
                x7 = torch.cat([x3, x7], dim=1)
            elif i == 11:
                x7 = layer(x7)
                x7 = torch.cat([x2, x7], dim=1)
            elif i == 12:
                x7 = layer(x7)
                x7 = torch.cat([x1, x7], dim=1)
            else:
                x7 = layer(x7)
                x7 = x7 + x
            i = i + 1

        return x7


class U_Netblock_6(nn.Module):
    def __init__(self, in_dim, out_dim, down=None, up=None, depths=12):
        super(U_Netblock_6, self).__init__()
        self.num = depths
        self.down = down
        self.up = up
        if self.down is not None:
            self.down = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, dilation=1, ceil_mode=False)
        if self.up is not None:
            # self.up = nn.ConvTranspose2d(in_channels= out_dim, out_channels= out_dim,kernel_size=4,stride=12,padding=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.block = nn.ModuleList()
        for i in range(self.num):
            if i == 0:
                blk = BasicLayer(in_dim=in_dim, out_dim=out_dim)
            else:
                blk = BasicLayer(in_dim=out_dim if 1 <= i <= 6 else 2 * out_dim, out_dim=out_dim,
                                 down=True if 2 <= i <= 6 else None, up=True if 6 <= i <= 10 else None)
            self.block.append(blk)

    def forward(self, x):
        i = 0
        if self.down is not None:
            x = self.down(x)
        for layer in self.block:
            if i == 0:
                x = layer(x)
            elif i == 1:
                x1 = layer(x)
            elif i == 2:
                x2 = layer(x1)
            elif i == 3:
                x3 = layer(x2)
            elif i == 4:
                x4 = layer(x3)
            elif i == 5:
                x5 = layer(x4)
            elif i == 6:
                x6 = layer(x5)
                x6 = torch.cat([x5, x6], dim=1)
            elif i == 7:
                x6 = layer(x6)
                x6 = torch.cat([x4, x6], dim=1)
            elif i == 8:
                x6 = layer(x6)
                x6 = torch.cat([x3, x6], dim=1)
            elif i == 9:
                x6 = layer(x6)
                x6 = torch.cat([x2, x6], dim=1)
            elif i == 10:
                x6 = layer(x6)
                x6 = torch.cat([x1, x6], dim=1)
            else:
                x6 = layer(x6)
                x6 = x6 + x
            i = i + 1
        if self.up is not None:
            x6 = self.up(x6)
        return x6


class U_Netblock_5(nn.Module):
    def __init__(self, in_dim, out_dim, down=None, up=None, depths=10):
        super(U_Netblock_5, self).__init__()
        self.num = depths
        self.down = down
        self.up = up
        if self.down is not None:
            self.down = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, dilation=1, ceil_mode=False)
        if self.up is not None:
            # self.up = nn.ConvTranspose2d(in_channels= out_dim, out_channels= out_dim,kernel_size=4,stride=12,padding=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.block = nn.ModuleList()
        for i in range(self.num):
            if i == 0:
                blk = BasicLayer(in_dim=in_dim, out_dim=out_dim)
            else:
                blk = BasicLayer(in_dim=out_dim if 1 <= i <= 5 else 2 * out_dim, out_dim=out_dim,
                                 down=True if 2 <= i <= 5 else False, up=True if 5 <= i <= 8 else False)
            self.block.append(blk)

    def forward(self, x):
        i = 0
        if self.down is not None:
            x = self.down(x)
        for layer in self.block:
            if i == 0:
                x = layer(x)
            elif i == 1:
                x1 = layer(x)
            elif i == 2:
                x2 = layer(x1)
            elif i == 3:
                x3 = layer(x2)
            elif i == 4:
                x4 = layer(x3)
            elif i == 5:
                x5 = layer(x4)
                x5 = torch.cat([x4, x5], dim=1)
            elif i == 6:
                x5 = layer(x5)
                x5 = torch.cat([x3, x5], dim=1)
            elif i == 7:
                x5 = layer(x5)
                x5 = torch.cat([x2, x5], dim=1)
            elif i == 8:
                x5 = layer(x5)
                x5 = torch.cat([x1, x5], dim=1)
            else:
                x5 = layer(x5)
                x5 = x5 + x
            i = i + 1
        if self.up is not None:
            x5 = self.up(x5)
        return x5


class DilatedBlock(nn.Module):
    def __init__(self, in_dim, out_dim, down=None, up=None):
        super(DilatedBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReplicationPad2d(padding=(1, 1, 1, 1)),
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
            nn.ReplicationPad2d(padding=(1, 1, 1, 1)),
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.ReplicationPad2d(padding=(1, 1, 1, 1)),
            nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
            nn.ReplicationPad2d(padding=(1, 1, 1, 1)),
            nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.ReplicationPad2d(padding=(2, 2, 2, 2)),
            nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, dilation=2, padding=0,
                      bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
            nn.ReplicationPad2d(padding=(2, 2, 2, 2)),
            nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, dilation=2, padding=0,
                      bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.ReplicationPad2d(padding=(4, 4, 4, 4)),
            nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, dilation=4, padding=0,
                      bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
            nn.ReplicationPad2d(padding=(4, 4, 4, 4)),
            nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, dilation=4, padding=0,
                      bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.ReplicationPad2d(padding=(8, 8, 8, 8)),
            nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, dilation=8, padding=0,
                      bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
            nn.ReplicationPad2d(padding=(8, 8, 8, 8)),
            nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, dilation=8, padding=0,
                      bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
        )
        self.conv6 = nn.Sequential(
            nn.ReplicationPad2d(padding=(4, 4, 4, 4)),
            nn.Conv2d(in_channels=2 * out_dim, out_channels=out_dim, kernel_size=3, stride=1, dilation=4, padding=0,
                      bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
            nn.ReplicationPad2d(padding=(4, 4, 4, 4)),
            nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, dilation=4, padding=0,
                      bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
        )
        self.conv7 = nn.Sequential(
            nn.ReplicationPad2d(padding=(2, 2, 2, 2)),
            nn.Conv2d(in_channels=2 * out_dim, out_channels=out_dim, kernel_size=3, stride=1, dilation=2, padding=0,
                      bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
            nn.ReplicationPad2d(padding=(2, 2, 2, 2)),
            nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, dilation=2, padding=0,
                      bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
        )
        self.conv8 = nn.Sequential(
            nn.ReplicationPad2d(padding=(1, 1, 1, 1)),
            nn.Conv2d(in_channels=2 * out_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
            nn.ReplicationPad2d(padding=(1, 1, 1, 1)),
            nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
        )
        self.down = down
        self.up = up
        if self.down is not None:
            self.down = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, dilation=1, ceil_mode=False)
        if self.up is not None:
            # self.up = nn.ConvTranspose2d(in_channels= out_dim, out_channels= out_dim,kernel_size=4,stride=12,padding=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        if self.down is not None:
            x = self.down(x)
        x = self.conv1(x)
        x1 = self.conv2(x)
        x2 = self.conv3(x1)
        x3 = self.conv4(x2)
        x4 = self.conv5(x3)
        x4 = torch.cat([x3, x4], dim=1)
        x4 = self.conv6(x4)
        x4 = torch.cat([x2, x4], dim=1)
        x4 = self.conv7(x4)
        x4 = torch.cat([x1, x4], dim=1)
        x4 = self.conv8(x4)
        x4 = x + x4
        if self.up is not None:
            x4 = self.up(x4)
        return x4


class Unwrapping(nn.Module):
    def __init__(self, depths=7, in_dims=[1, 64, 128, 256, 512, 256, 128], out_dims=[64, 128, 256, 256, 128, 64, 64]):
        super(Unwrapping, self).__init__()
        self.depths = depths
        self.block = nn.ModuleList()
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        for i in range(self.depths):
            if i == 0 or i == 6:
                blk = U_Netblock_7(in_dim=in_dims[i], out_dim=out_dims[i])
            elif i == 1 or i == 5:
                blk = U_Netblock_6(in_dim=in_dims[i], out_dim=out_dims[i], down=True if i == 1 else None,
                                   up=True if i == 5 else None)
            elif i == 2 or i == 4:
                blk = U_Netblock_5(in_dim=in_dims[i], out_dim=out_dims[i], down=True if i == 2 else None,
                                   up=True if i == 4 else None)
            else:
                blk = DilatedBlock(in_dim=in_dims[i], out_dim=out_dims[i], down=True, up=True)
            self.block.append(blk)

    def forward(self, x):
        i = 0
        for layer in self.block:
            if i == 0:
                x = layer(x)
            elif i == 1:
                x1 = layer(x)
            elif i == 2:
                x2 = layer(x1)
            elif i == 3:
                x3 = layer(x2)
                x3 = torch.cat([x2, x3], dim=1)
            elif i == 4:
                x3 = layer(x3)
                x3 = torch.cat([x1, x3], dim=1)
            elif i == 5:
                x3 = layer(x3)
                x3 = torch.cat([x, x3], dim=1)
            else:
                x3 = self.act(layer(x3))
                x3 = self.conv(x3)
            i = i + 1
        return x3


class New_Unwrapping(nn.Module):
    def __init__(self, depths=7, in_dims=[1, 64, 128, 256, 512, 256, 128], out_dims=[64, 128, 256, 256, 128, 64, 64]):
        super(New_Unwrapping, self).__init__()
        self.depths = depths
        self.block = nn.ModuleList()
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.refine = Refine()

        for i in range(self.depths):
            if i == 0 or i == 6:
                blk = U_Netblock_7(in_dim=in_dims[i], out_dim=out_dims[i])
            elif i == 1 or i == 5:
                blk = U_Netblock_6(in_dim=in_dims[i], out_dim=out_dims[i], down=True if i == 1 else None,
                                   up=True if i == 5 else None)
            elif i == 2 or i == 4:
                blk = U_Netblock_5(in_dim=in_dims[i], out_dim=out_dims[i], down=True if i == 2 else None,
                                   up=True if i == 4 else None)
            else:
                blk = DilatedBlock(in_dim=in_dims[i], out_dim=out_dims[i], down=True, up=True)
            self.block.append(blk)

    def forward(self, x):
        i = 0
        for layer in self.block:
            if i == 0:
                x = layer(x)
            elif i == 1:
                x1 = layer(x)
            elif i == 2:
                x2 = layer(x1)
            elif i == 3:
                x3 = layer(x2)
                x3 = torch.cat([x2, x3], dim=1)
            elif i == 4:
                x3 = layer(x3)
                x3 = torch.cat([x1, x3], dim=1)
            elif i == 5:
                x3 = layer(x3)
                x3 = torch.cat([x, x3], dim=1)
            else:
                x3 = self.act(layer(x3))
                x3 = self.refine(x3)
            i = i + 1
        return x3


class Refine(nn.Module):
    def __init__(self, in_dim=64, out_dim=64):
        super(Refine, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReplicationPad2d(padding=(1, 1, 1, 1)),
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.ReplicationPad2d(padding=(1, 1, 1, 1)),
            nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.ReplicationPad2d(padding=(2, 2, 2, 2)),
            nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, dilation=2, padding=0,
                      bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.ReplicationPad2d(padding=(4, 4, 4, 4)),
            nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, dilation=4, padding=0,
                      bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.ReplicationPad2d(padding=(8, 8, 8, 8)),
            nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, dilation=8, padding=0,
                      bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            self.conv1,
            self.conv3,
            self.conv4,
            self.conv5,
            self.conv2
        )
        self.branch = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.act = nn.LeakyReLU(inplace=True)
        self.conv_last = nn.Sequential(
            nn.ReplicationPad2d(padding=(1, 1, 1, 1)),
            nn.Conv2d(in_channels=out_dim, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False))

    def forward(self, x):
        x1 = x
        x1 = self.branch(x1)
        x = self.conv(x)
        x = self.act(x + x1)
        x = self.conv_last(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReplicationPad2d(padding=(1, 1, 1, 1)),
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
            nn.ReplicationPad2d(padding=(1, 1, 1, 1)),
            nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dim)
        )
        self.branch = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1, padding=0,
                                bias=True)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        branch = self.branch(x)
        x = self.res_block(x)
        x = self.act(x + branch)
        return x


class Refine1(nn.Module):
    def __init__(self, in_dim=64, out_dim=64, depths=4):
        super(Refine1, self).__init__()
        self.depths = depths
        self.block = nn.ModuleList()
        for i in range(self.depths):
            blk = ResBlock(in_dim=in_dim, out_dim=out_dim)
            self.block.append(blk)
        self.branch = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.last_conv = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.ReplicationPad2d(padding=(1, 1, 1, 1)),
            nn.Conv2d(in_channels=out_dim, out_channels=1, kernel_size=3, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        branch = self.branch(x)
        for blk in self.block:
            x = blk(x)
        x = self.last_conv(x + branch)
        return x


class ResUnwrapping(nn.Module):
    def __init__(self, depths=7, in_dims=[1, 64, 128, 256, 512, 256, 128], out_dims=[64, 128, 256, 256, 128, 64, 64]):
        super(ResUnwrapping, self).__init__()
        self.depths = depths
        self.block = nn.ModuleList()
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.refine = Refine1()

        for i in range(self.depths):
            if i == 0 or i == 6:
                blk = U_Netblock_7(in_dim=in_dims[i], out_dim=out_dims[i])
            elif i == 1 or i == 5:
                blk = U_Netblock_6(in_dim=in_dims[i], out_dim=out_dims[i], down=True if i == 1 else None,
                                   up=True if i == 5 else None)
            elif i == 2 or i == 4:
                blk = U_Netblock_5(in_dim=in_dims[i], out_dim=out_dims[i], down=True if i == 2 else None,
                                   up=True if i == 4 else None)
            else:
                blk = DilatedBlock(in_dim=in_dims[i], out_dim=out_dims[i], down=True, up=True)
            self.block.append(blk)

    def forward(self, x):
        i = 0
        for layer in self.block:
            if i == 0:
                x = layer(x)
            elif i == 1:
                x1 = layer(x)
            elif i == 2:
                x2 = layer(x1)
            elif i == 3:
                x3 = layer(x2)
                x3 = torch.cat([x2, x3], dim=1)
            elif i == 4:
                x3 = layer(x3)
                x3 = torch.cat([x1, x3], dim=1)
            elif i == 5:
                x3 = layer(x3)
                x3 = torch.cat([x, x3], dim=1)
            else:
                x3 = self.act(layer(x3))
                x3 = self.refine(x3)
            i = i + 1
        return x3


