import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.tools import extract_image_patches, \
    reduce_mean, reduce_sum, same_padding

def default_conv(in_channels, out_channels, kernel_size, group_num=1, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, groups=group_num,
        padding=(kernel_size//2), stride=stride, bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4025, 0.4482, 0.4264), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)

        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
        bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.PReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res
    ########################################################################################################################
    ########################################################################################################################

class attention_block(nn.Module):
    def __init__(self, n_feats, patchsize, padd, res_scale=1):
        super(attention_block, self).__init__()
        self.res_scale = res_scale

        self.atten1 = MultiscaleAttentionA(in_channel=n_feats, res_scale=res_scale, kernel_size=patchsize[0], padding=padd[0])
        self.atten2 = MultiscaleAttentionA(in_channel=n_feats, res_scale=res_scale, kernel_size=patchsize[1], padding=padd[1])
        self.atten3 = MultiscaleAttentionA(in_channel=n_feats, res_scale=res_scale, kernel_size=patchsize[2], padding=padd[2])
        # self.atten4 = MultiscaleAttentionA(in_channel=n_feats, res_scale=res_scale, kernel_size=patchsize[3],padding=padd[3])
        self.fusion = default_conv(in_channels=n_feats*3, out_channels=n_feats, kernel_size=1, bias=True)
        self.CA = CAer(n_feats)
    def forward(self, x):
        res = x

        a1 = self.atten1(x)
        a2 = self.atten2(x)
        a3 = self.atten3(x)
        # a4 = self.atten4(x)
        a = torch.cat([a1, a2, a3], 1)
        # a = torch.cat([a1, a2], 1)
        output = self.fusion(a)
        output = self.CA(output)
        res = res + output
        return res
        # return res, a1, a2, a3    ##此处加了a123
    #####################################################################################################################

class My_Block(nn.Module):
    def __init__(self, conv, n_feats, bias=True, act=nn.GELU(), res_scale=1):
        super(My_Block, self).__init__()
        self.res_scale = res_scale
        ms_body1 = []
        # ms_body1.append(conv(n_feats, n_feats, 1, bias=bias))
        # ms_body1.append(conv(n_feats, n_feats, 3, group_num=n_feats, bias=bias))
        ms_body1.append(conv(n_feats, n_feats, 3, bias=bias))
        ms_body1.append(act)

        ms_body2 = []
        # ms_body2.append(conv(n_feats, n_feats, 1, bias=bias))
        # ms_body2.append(conv(n_feats, n_feats, 3, group_num=n_feats, bias=bias))
        ms_body2.append(conv(n_feats, n_feats, 3, bias=bias))
        ms_body2.append(act)

        ms_body3=[]
        # ms_body3.append(conv(n_feats, n_feats, 1, bias=bias))
        # ms_body3.append(conv(n_feats, n_feats, 3, group_num=n_feats, bias=bias))
        ms_body3.append(conv(n_feats, n_feats, 3, bias=bias))
        ms_body3.append(act)

        self.branch1 = nn.Sequential(*ms_body1)
        self.branch2 = nn.Sequential(*ms_body2)
        self.branch3 = nn.Sequential(*ms_body3)
        self.fusion = conv(n_feats * 3, n_feats, 1, bias=bias)
        # self.act = act
        # self.CA = CAer(n_feats)
        # self.esa = ESA(n_feats, nn.Conv2d)
        self.CCA = CCALayer(n_feats)

    def forward(self, x):
        res = x
        x1 = self.branch1(x)
        x2 =self.branch2(x1)
        x3 =self.branch3(x2)
        bag = torch.cat([x1, x2, x3], dim=1)
        bag1 = self.fusion(bag)   # 1*1 Conv
        out = self.CCA(bag1)
        # out = self.esa(out1)
        output = res + out
        # output = out.mul(self.res_scale)
        # output = self.fusion(output)
        # output = self.act(output)

        # res = self.SA(res)*self.res_scale+res
        return output

class MS_Block(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True,  act=nn.PReLU(), res_scale=1):
        super(MS_Block, self).__init__()
        self.res_scale = res_scale
        m1 = []
        m1.append(conv(n_feats, n_feats, 1, bias=bias))
        m1.append(act)
        m1.append(conv(n_feats, n_feats, kernel_size, bias=bias))
        m1.append(act)
        m1.append(conv(n_feats, n_feats, kernel_size, bias=bias))

        m2 = []
        m2.append(conv(n_feats, n_feats, 1, bias=bias))
        m2.append(act)
        m2.append(conv(n_feats, n_feats, kernel_size, bias=bias))

        m3 = []
        m3.append(conv(n_feats, n_feats, 1, bias=bias))
        m3.append(act)
        self.branch1 =nn.Sequential(*m1)
        self.branch2 = nn.Sequential(*m2)
        self.branch3 = nn.Sequential(*m3)
        self.CA = CAer(n_feats*3)
        self.fusion = conv(n_feats*3, n_feats, 1, bias=bias)
        self.fusion2 =conv(n_feats*3, n_feats, 1, bias=bias)
        # self.SA = MultiscaleAttentionA(n_feats)
    def forward(self, x):
        res = x
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x = torch.cat([x1, x2, x3], 1)
        out = self.CA(x)
        output = self.fusion(out)
        output = output.mul(self.res_scale)
        res = res+output
        return res

def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.GELU = nn.GELU()

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)  # k_s benlai shi 7
        v_range = self.GELU(self.conv_max(v_max))
        c3 = self.GELU(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m

class CAer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CAer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )


    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x*y

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class MultiscaleAttentionA(nn.Module):
    def __init__(self,  in_channel, res_scale=1, kernel_size=3,padding=1):
        super(MultiscaleAttentionA, self).__init__()
        self.scale = res_scale
        self.res_scale = res_scale
        self.down_sample = nn.Sequential(
            *[nn.Conv2d(in_channel, in_channel, 3, stride=1, padding=kernel_size//2), nn.PReLU()])
        # self.down_sample = nn.Sequential(
        #     *[nn.Conv2d(in_channel, in_channel, kernel_size, stride=1, padding=kernel_size // 2), nn.PReLU()])

        self.stoL = CrossScaleAttentionA(channel=in_channel, res_scale=self.res_scale, ksize=kernel_size, padding=padding)
        # self.stoL = CrossScaleAttentionA(channel=in_channel, res_scale=self.res_scale)
    def forward(self, input):
        res = input

        s = self.down_sample(input)
        res = res + (self.stoL(res, s)).mul(self.res_scale)

        return res


class CrossScaleAttentionA(nn.Module):
    def __init__(self, channel=256, res_scale=1, reduction=2, ksize=3, padding=1, scale=2, stride=1, softmax_scale=10, average=True,
                 conv=default_conv):
        super(CrossScaleAttentionA, self).__init__()
        self.ksize = ksize
        self.padding = padding
        self.stride = stride
        self.softmax_scale = softmax_scale
        self.res_scale = res_scale
        ##修改channel=128->156;
        self.scale = scale
        self.average = average
        escape_NaN = torch.FloatTensor([1e-4])
        self.register_buffer('escape_NaN', escape_NaN)
        self.conv_match_1 = BasicBlock(conv, channel, channel//reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match_2 = BasicBlock(conv, channel, channel//reduction, 1, bn=False, act=nn.PReLU())
        self.conv_assembly = BasicBlock(conv, channel, channel, 1, bn=False, act=nn.PReLU())
        # self.register_buffer('fuse_weight', fuse_weight)

    def forward(self, inputL, inputs):
        res = inputL
        #用小表示大，浅层表示深层
        # get embedding 先将其映射后,后作为卷积核
        embed_w = self.conv_assembly(inputs)       #作为反卷积核
        kernel = self.ksize
        padd = self.padding
        shape_input = list(embed_w.size())  # b*c*h*w
        # raw_w is extracted for reconstruction  提取反卷积核
        raw_w = extract_image_patches(embed_w, ksizes=[kernel, kernel],
                                      strides=[self.stride, self.stride],
                                      rates=[1, 1],
                                      padding='same')  # [N, C*k*k, L]
        raw_w = raw_w.view(shape_input[0], shape_input[1], kernel, kernel, -1)  # raw_shape: [N, C, k, k, L]
        raw_w = raw_w.permute(0, 4, 1, 2, 3)  # raw_shape: [N, L, C, k, k]
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # 制作卷积核
        ref = self.conv_match_2(inputs)
                #从小图中提取卷积核
        w = extract_image_patches(ref, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        shape_ref = ref.shape  # 1,128,12,12
        # w shape: [N, C, k, k, L]
        w = w.view(shape_ref[0], shape_ref[1], self.ksize, self.ksize, -1)
        w = w.permute(0, 4, 1, 2, 3)  # w shape: [N, L, C, k, k]
        w_groups = torch.split(w, 1, dim=0)
        #作为中间路输入
        match_input = self.conv_match_1(inputL)  # 作为输入
        input_groups = torch.split(match_input, 1, dim=0)

        y = []
        scale = self.softmax_scale
        # 1*1*k*k
        # fuse_weight = self.fuse_weight

        for xi, wi, raw_wi in zip(input_groups, w_groups, raw_w_groups):
            # normalize
            wi = wi[0]  # [L, C, k, k]
            max_wi = torch.max(torch.sqrt(reduce_sum(torch.pow(wi, 2),
                                                     axis=[1, 2, 3],
                                                     keepdim=True)),
                               self.escape_NaN)
            wi_normed = wi / max_wi             #144,128,3,3

            # Compute correlation map
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)  # [1, L, H, W] L = shape_ref[2]*shape_ref[3]

            # yi = yi.view(1, shape_ref[2] * shape_ref[3], 24, 24)  # (B=1, C=32*32, H=32, W=32)
            # rescale matching score
            yi = F.softmax(yi * scale, dim=1)
            if self.average == False:
                yi = (yi == yi.max(dim=1, keepdim=True)[0]).float()

            # deconv for reconsturction
            wi_center = raw_wi[0]
            #yi = F.conv_transpose2d(yi, wi_center, stride=self.stride * self.scale, padding=self.scale)
            yi = F.conv_transpose2d(yi, wi_center, stride=self.stride, padding=padd)

            y.append(yi)

        # y = torch.cat(y, dim=0)
        y = torch.cat(y, dim=0) + res * self.res_scale
        return y
