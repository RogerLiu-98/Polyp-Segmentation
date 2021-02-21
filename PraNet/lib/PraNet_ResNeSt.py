import torch, torchvision
import math
from resnest.torch.resnest import resnest50
import torch.nn as nn
import torch.nn.functional as F



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class RFB(nn.Module):
    # RFB-like multi-scale module
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=5, padding=2),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=7, padding=3),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = x_cat + self.conv_res(x)
        return x


class IDA(nn.Module):
    def __init__(self, node_kernel, out_dim, channels, up_factors):
        super(IDA, self).__init__()
        self.channels = channels
        self.out_dim = out_dim
        for i, c in enumerate(channels):
            if c == out_dim:
                proj = nn.Identity()
            else:
                proj = BasicConv2d(c, out_dim, kernel_size=1, stride=1, bias=False)
            f = int(up_factors[i])
            if f == 1:
                up = nn.Identity()
            else:
                up = nn.Upsample(scale_factor=f, mode='bilinear', align_corners=True)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)

        for i in range(1, len(channels)):
            node = BasicConv2d(out_dim * 2, out_dim, kernel_size=node_kernel, stride=1, padding=node_kernel // 2,
                               bias=False)
            setattr(self, 'node_' + str(i), node)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.decode = nn.Conv2d(out_dim, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, layers):
        assert len(self.channels) == len(layers), \
            '{} vs {} layers'.format(len(self.channels), len(layers))
        layers = list(layers)
        for i, l in enumerate(layers):
            upsample = getattr(self, 'up_' + str(i))
            project = getattr(self, 'proj_' + str(i))
            layers[i] = upsample(project(l))
        x = layers[0]
        for i in range(1, len(layers)):
            node = getattr(self, 'node_' + str(i))
            x = node(torch.cat([x, layers[i]], 1))
        x = self.decode(x)
        return x


class MSCAM(nn.Module):

    def __init__(self, in_channels, r=4):
        super(MSCAM, self).__init__()
        inter_channels = in_channels // r
        self.local_att = nn.Sequential(
            BasicConv2d(in_channels, inter_channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels)
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            BasicConv2d(in_channels, inter_channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = torch.sigmoid(xlg)
        return wei


class iAFF(nn.Module):

    def __init__(self, in_channels, r=4):
        super(iAFF, self).__init__()
        self.ms_cam1 = MSCAM(in_channels, r)
        self.ms_cam2 = MSCAM(in_channels, r)

    def forward(self, x, y):
        x1 = x + y
        x1l = self.ms_cam1(x1)
        x1r = 1 - x1l
        x2 = x * x1l + y * x1r

        x2l = self.ms_cam2(x2)
        x2r = 1 - x2l
        z = x * x2l + y * x2r
        return z


class Decode(nn.Module):

    def __init__(self, in_channels, r=4):
        super(Decode, self).__init__()
        inter_channels = in_channels // r
        self.decode1 = BasicConv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.decode2 = BasicConv2d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.decode3 = BasicConv2d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.decode4 = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.decode1(x)
        x = self.decode2(x)
        x = self.decode3(x)
        x = self.decode4(x)

        return x


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # ResNeSt
        self.resnest = resnest50(pretrained=True)

        # RFA
        self.rfa3 = RFB(512, 32)
        self.rfa4 = RFB(1024, 32)
        self.rfa5 = RFB(2048, 32)

        # self.aggregation = Aggregation(32)
        self.aggregation = IDA(node_kernel=3, out_dim=32, channels=[32, 32, 32], up_factors=[1, 2, 4])

        # IAFF
        self.iaff5 = iAFF(in_channels=32, r=2)
        self.iaff4 = iAFF(in_channels=32, r=2)
        self.iaff3 = iAFF(in_channels=32, r=2)

        # Projection
        self.proj1 = BasicConv2d(1, 32, kernel_size=1, stride=1, padding=0, bias=True)
        self.proj2 = BasicConv2d(1, 32, kernel_size=1, stride=1, padding=0, bias=True)
        self.proj3 = BasicConv2d(1, 32, kernel_size=1, stride=1, padding=0, bias=True)

        self.dcd1 = Decode(32, r=2)
        self.dcd2 = Decode(32, r=2)
        self.dcd3 = Decode(32, r=2)

        # Up-Sample and Down-Sample
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down = nn.Upsample(scale_factor=.25, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Gate
        f1 = self.resnest.conv1(x)
        f1 = self.resnest.bn1(f1)
        f1 = self.resnest.relu(f1)
        f1 = self.resnest.maxpool(f1)

        # Layer1
        f2 = self.resnest.layer1(f1)

        # layer2
        f3 = self.resnest.layer2(f2)

        # layer3
        f4 = self.resnest.layer3(f3)

        # layer4
        f5 = self.resnest.layer4(f4)

        # RFA
        f3_rfa = self.rfa3(f3)
        f4_rfa = self.rfa4(f4)
        f5_rfa = self.rfa5(f5)

        # Aggregation
        global_map = self.aggregation([f3_rfa, f4_rfa, f5_rfa])
        out_4 = F.interpolate(global_map, scale_factor=8, mode='bilinear', align_corners=True)

        # iAFF5
        s6 = self.down(global_map)
        r5 = self.iaff5(f5_rfa, self.proj1(s6))
        r5 = self.dcd1(r5)
        s5 = r5 + s6
        out_3 = F.interpolate(s5, scale_factor=32, mode='bilinear', align_corners=True)

        # iAFF4
        s5 = self.up(r5)
        r4 = self.iaff4(f4_rfa, self.proj2(s5))
        r4 = self.dcd2(r4)
        s4 = r4 + s5
        out_2 = F.interpolate(s4, scale_factor=16, mode='bilinear', align_corners=True)

        # iAFF3
        s4 = self.up(r4)
        r3 = self.iaff3(f3_rfa, self.proj3(s4))
        r3 = self.dcd3(r3)
        s3 = r3 + s4
        out_1 = F.interpolate(s3, scale_factor=8, mode='bilinear', align_corners=True)

        return out_4, out_3, out_2, out_1


if __name__ == '__main__':
    # x = torch.randn(4, 3, 512, 512).requires_grad_(True)
    # net = Net()
    # y = net(x)
    #
    # ConvNetVis = make_dot(y, params=dict(list(net.named_parameters()) + [('input', x)]))
    # ConvNetVis.format = 'png'
    # ConvNetVis.directory = '/home/roger/PycharmProjects/PraNet-Pytorch'
    # ConvNetVis.view()
    import hiddenlayer as h
    net = Net()
    vis_graph = h.build_graph(net, torch.randn(4, 3, 512, 512))
    vis_graph.theme = h.graph.THEMES["blue"].copy()
    vis_graph.save("./demo1.png")