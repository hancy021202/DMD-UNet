import torch.nn as nn
import numpy as np
import torch



class Unwrapping(nn.Module):
    def __init__(self, num_blocks=[2, 2, 3, 3, 3, 3, 3, 3, 3, 2, 2],
                 dim_in=[64, 64, 128, 256, 512, 512, 1024, 1024, 512, 256, 128],
                 dim_out=[64, 128, 256, 512, 512, 512, 512, 256, 128, 64, 64]):
        super(Unwrapping, self).__init__()
        self.depths = len(num_blocks)
        self.layers = nn.ModuleList()
        for i_layer in range(self.depths):
            layer = BasicLayer(num_block=num_blocks[i_layer], in_dim=dim_in[i_layer], out_dim=dim_out[i_layer],
                               down=True if 0 < i_layer < 6 else False, up=True if 4 < i_layer < 10 else False)
            self.layers.append(layer)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        i = 0
        for layer in self.layers:
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
                x5 = torch.cat([x4, x5], dim=1)
                x5 = layer(x5)
            elif i == 7:
                x5 = torch.cat([x3, x5], dim=1)
                x5 = layer(x5)
            elif i == 8:
                x5 = torch.cat([x2, x5], dim=1)
                x5 = layer(x5)
            elif i == 9:
                x5 = torch.cat([x1, x5], dim=1)
                x5 = layer(x5)
            else:
                x5 = torch.cat([x, x5], dim=1)
                x5 = self.conv2(layer(x5))
            i += 1
        return x5

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding,
                                        groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=True, switch=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.nonlinearity = nn.ReLU()
        self.switch = switch
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups,
                                         bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.LayerNorm(480) \
            # self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) \
            # if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                   stride=stride, padding=0, groups=groups)

    def forward(self, inputs):
        if self.switch:
            self.switch_to_deploy()
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True




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


class BasicLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_block, down=False, up=False):
        super(BasicLayer, self).__init__()
        self.num = num_block
        self.down = down
        self.up = up
        self.conv = nn.Conv2d(in_channels=in_dim,out_channels=out_dim,kernel_size=1,stride=1,padding=0,bias=True)
        if self.down is True:
            self.down = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, dilation=1, ceil_mode=False)
        if self.up is True:
            #self.up = nn.ConvTranspose2d(in_channels= out_dim, out_channels= out_dim,kernel_size=4,stride=12,padding=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.block = nn.ModuleList()
        for i in range(self.num):
            if i == 0:
                blk = RepVGGBlock(in_channels=in_dim, out_channels=out_dim)
            else:
                blk = RepVGGBlock(in_channels=out_dim, out_channels=out_dim)
            self.block.append(blk)

    def forward(self, x):
        if self.down is not False:
            x = self.down(x)
        x1 = x
        x1 = self.conv(x)
        for blk in self.block:
            x = blk(x)
        x = x1 + x
        if self.up is not False:
            x = self.up(x)
        return x
