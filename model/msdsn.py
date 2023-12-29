import torch

from model import common
# import BSRN_arch as BSRN
# import block as block
# from model import cmsnl
import torch.nn as nn
# from model import myattention
from thop import profile
from option import args

def make_model(args, parent=False):
    if args.dilation:
        from model import dilated
        return MSDN(args, dilated.dilated_conv)
    else:
        return MSDN(args)

class My_ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, act, res_scale, n_resblocks):
        super(My_ResidualGroup, self).__init__()
        modules_body = [
            common.My_Block(
                conv, n_feat, bias=True, res_scale=res_scale)
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, 3))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class MSDN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(MSDN, self).__init__()
        n_resgroups = args.n_resgroups
        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act1 = nn.PReLU()

        patch_size = [3, 5, 7, 9]
        paddings = [1, 2, 3, 4]

        rgb_mean = (0.4025, 0.4482, 0.4264)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.fusion = nn.Conv2d(n_feats * n_resgroups//2, n_feats, 1, groups=1, padding=0, stride=1, bias=True)
        fusion_list = [
            nn.Conv2d(n_feats * 2, n_feats, 1, groups=1, padding=0, stride=1, bias=True) for _ in range(n_resgroups//2 - 1)]
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        m_body = [
            My_ResidualGroup(
                conv, n_feats,  act=act1, res_scale=args.res_scale, n_resblocks=n_resblock
            ) for _ in range(n_resgroups//2)
        ]

        m_body.append(common.attention_block(n_feats=n_feats, patchsize=patch_size, padd=paddings,
                                             res_scale=args.res_scale))

        for _ in range(n_resgroups//2):
            m_body.append(My_ResidualGroup(
                conv, n_feats,  act=act1, res_scale=args.res_scale, n_resblocks=n_resblock
            ))

        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            nn.Conv2d(
                n_feats, args.n_colors, kernel_size,
                padding=(kernel_size//2)
            )
        ]
        # self.upsample = nn.ModuleList([common.Upsampler(conv, scale, n_feats, act=False)])
        # m_tail = [conv(n_feats, args.n_colors, kernel_size, padding=kernel_size // 2)]
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.num_group = args.n_resgroups
        self.fusion_list = nn.Sequential(*fusion_list)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = x
        group_out = []
        for i in range(self.num_group//2):
            x = self.body[i](x)
            group_out.append(x)
        half_group_out = torch.cat(group_out, 1)
        half_group_out1 = self.fusion(half_group_out)

        msnlab_out = self.body[self.num_group//2](half_group_out1)

        for i in range(self.num_group//2 + 1, self.num_group//2 * 2 + 1):
            if i == self.num_group//2 + 1:
                post_half_group_input = msnlab_out
            else:
                concate = torch.cat((post_half_group_input, msnlab_out), 1)
                post_half_group_input = self.fusion_list[i - self.num_group//2 - 2](concate)

            post_half_group_out = self.body[i](post_half_group_input)
            post_half_group_input = post_half_group_out

        post_half_group_out = self.body[self.num_group//2 * 2 + 1](post_half_group_out)
        res = res + post_half_group_out
        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


if __name__ == '__main__':
    device = torch.device("cuda:1")
    model = MSDN(args)
    _model = model.to(device)
    input = torch.randn(1, 3, 48, 48).to(device)
    # input.to(torch.device('cuda'))
    print(input.shape)
    flops, params = profile(_model, inputs=(input, ))
    print('params:', params/10**6)
    print('flops:', flops/10**9)

