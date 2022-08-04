from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
import torch

class B_relu_Conv(nn.Module):
    def __init__(self, dim, output):
        super(B_relu_Conv, self).__init__()
        self.bn = nn.BatchNorm3d(dim)
        self.relu = nn.LeakyReLU()
        self.conv = nn.Conv3d(dim, output, kernel_size=5, padding="same")
    def forward (self,src):
        bn = self.bn(src)
        relu = self.relu(bn)
        conv = self.conv(relu)
        return conv


class ResBlock(nn.Module):
    def __init__(self, dim, output):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv3d(dim, output, kernel_size=5, padding="same")
        self.conv1 = B_relu_Conv(output, output)
        self.conv2 = B_relu_Conv(output, output)
    def forward (self, src):
        conv = self.conv(src)
        conv = self.conv1(conv)
        conv = self.conv2(conv)
        return conv
class V_net(nn.Module):
    def __init__(self):
        super(V_net, self).__init__()
        self.conv1 = ResBlock(1, 4)
        self.pool1 = nn.MaxPool3d((1,2,2))
        self.conv2 = ResBlock(4, 8)
        self.pool2 = nn.MaxPool3d(2)
        self.conv3 = ResBlock(8, 16)
        self.pool3 = nn.MaxPool3d((1,2,2))
        self.conv4 = ResBlock(16, 32)
        self.pool4 = nn.MaxPool3d(2)
        self.conv5 = ResBlock(32, 64)

        self.up1 = nn.Upsample((64,32,32))
        self.up1_ = B_relu_Conv(64, 4)
        self.deconv1 = ResBlock(36, 32)
        #concat
        self.up2 = nn.Upsample((64,64,64))
        self.up2_ = B_relu_Conv(32, 4)
        self.deconv2 = ResBlock(20, 16)
        #concat
        self.up3 = nn.Upsample((128,128,128))
        self.up3_ = B_relu_Conv(16, 4)
        self.deconv3 = ResBlock(12, 8)
        #concat
        self.up4 = nn.Upsample((128,256,256))
        self.up4_ = B_relu_Conv(8, 4)
        self.deconv4 = ResBlock(8, 4)
        
        self.convout = nn.Conv3d(4, 1, kernel_size=5, padding="same")
    def forward(self, src):
        conv1 = self.conv1(src)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        bottom = self.conv5(pool4)
        up = self.up1(bottom)
        up = self.up1_(up)
        merge = torch.cat((up,conv4), dim=1)
        deconv1 = self.deconv1(merge)
        up1 = self.up2(deconv1)
        up1 = self.up2_(up1)
        merge = torch.cat([up1,conv3], axis=1)
        deconv2 = self.deconv2(merge)
        up2 = self.up3(deconv2)
        up2 = self.up3_(up2)
        merge = torch.cat([up2,conv2], axis=1)
        deconv3 = self.deconv3(merge)
        up3 = self.up4(deconv3)
        up3 = self.up4_(up3)
        merge = torch.cat([up3,conv1], axis=1)
        deconv4 = self.deconv4(merge)
        return self.convout(deconv4)
