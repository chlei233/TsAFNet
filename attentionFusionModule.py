import torch.nn as nn
import torch
import torchvision

class AFLayerBasic2d(nn.Module):
    # base on the seNet codebase
    # 2d apply with tsn base config, weight the softmax score
    def __init__(self, segment, in_channels):
        super(AFLayerBasic2d, self).__init__()
        
        self.segment = segment

        self.avg_pool = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.segment, self.segment//2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.segment//2, self.segment, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        nt, _, _, _ = x.size()
        b_batch = nt // self.segment
        y = self.avg_pool(x).view(b_batch, self.segment)
        #print(y[2])
        #print(y.shape)
        y = self.fc(y)
        return y

class AFLayerBasic3d(nn.Module):
    # 3d apply for 3d conv
    def __init__(self, segment, in_channels):
        super(AFLayerBasic3d, self).__init__()

        self.segment = segment

        self.avg_pool = nn.Sequential(
            nn.Conv3d(in_channels, 1, 1),
            nn.AdaptiveAvgPool3d((self.segment, 1, 1))
        )

        self.fc = nn.Sequential(
            nn.Linear(self.segment, self.segment, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.segment, self.segment, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x.shape: b*c*t*h*w
        n, _, t, _, _ = x.size()
        y = self.avg_pool(x).view(n, t)
        #print(y[2])
        #print(y.shape)
        y = self.fc(y).view(n, 1, t, 1, 1)
        #print(y[2])
        return x * y.expand_as(x)

class AFLayerAtt(nn.Module):
    def __init__(self, segment, in_channels, temperature, att_dropout=0.1, posEncode=None, embed=None):
        super(AFLayerAtt, self).__init__()
        self.segment = segment

        self.posEncode = posEncode
        self.embed = embed
        self.temperature = temperature
        self.dropout = nn.Dropout(att_dropout)

        self.avg_pool = nn.Sequential(
            nn.Conv3d(in_channels, in_channels//2, 1),
            nn.AdaptiveAvgPool3d((self.segment, 1, 1))  # 与 non_local不同的是，这里获取和建模global的上下文信息  b*c*t*1*1
        )  #  晚上继续，总觉得有些东西可以被发现。。关于三个矩阵，或者三个值

        self.Q = torch.randn()
        pass








x = torch.randn(256, 32, 112, 112)
h = AFLayerBasic2d(4, 32)
c = AFLayerBasic3d(4, 32)
x_1 = torch.randn(64, 32, 4, 112, 112)
y = h(x)
g = c(x_1)
print(g.shape)
print(y.shape)

