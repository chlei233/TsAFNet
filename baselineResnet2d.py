"resnet block"
import torch
import torch.nn as nn
from attentionFusionModule import AFLayerBasic2d

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):  # for more 34 layers' net
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        # 保证残差输入的size同网络输出（第二个relu()函数之前）的size相同
        # 源码复用性更好，考虑也更周全
        self.downsample = downsample
        self.stride =stride
        '''if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride'''

    def forward(self, x):

        #residual = self.downsample(x)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)

        return out

class BasicBlockPreAct(BasicBlock):  # 2d plan 破产(temporary)
    def __init__(self, inplanes, planes, segment, stride=1, downsample=None): 
        super(BasicBlockPreAct, self).__init__(inplanes, planes, stride)
        self.af = AFLayerBasic2d(segment, inplanes)
    
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        y = self.af(out).regist
        
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4  # 详见resnet网络结构，第二个1x1层需要expansion倍的卷积核以(扩展逼近能力)并减少参数。。。conflict
 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        """if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x"""
        self.stride = stride
 
    def forward(self, x):
        #residual = self.downsample(x)
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
  
        out += residual
        out = self.relu(out)
 
        return out

class ResNet(nn.Module):
 
    def __init__(self, block, layers, num_classes=1000):  
        super(ResNet, self).__init__()
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,  # in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
 
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
 #  two different layers for lazy ?
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample)) 
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks): # 第一个block已经规范了residual的输入格式
            layers.append(block(self.inplanes, planes))
 
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
 
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 按batch number进行全连接操作
        x = self.fc(x)
        
        return x

class PreActResNet(ResNet):
    def __init__(self, block_style, blocks_num_list, num_classes=20):
        super(PreActResNet, self).__init__(block_style, blocks_num_list, num_classes)
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3, bias=False)  # crop to 224*224: (224+6-7+1)/2=112
        self.bn1 = nn.BatchNorm2d(16)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # out to 56

        self.layer1 = self._make_layer(block_style, 16, blocks_num_list[0])
        self.layer2 = self._make_layer(block_style, 32, blocks_num_list[1], stride=2)
        self.layer3 = self._make_layer(block_style, 64, blocks_num_list[2], stride=2)
        self.layer4 = self._make_layer(block_style, 128, blocks_num_list[3], stride=2)

        self.fc = nn.Linear(128 * block_style.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
 
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 按batch number进行全连接操作
        x = self.fc(x)
        
        return x

    """def _make_layer(self, block_style, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)"""

def resnet18():
    """Constructs a ResNet-18 model."""
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    return model
 
def preActResNet18():
    model = PreActResNet(BasicBlock, [2, 2, 2, 2])
    return model

def resnet34():
    """Constructs a ResNet-34 model."""
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    return model
 
 
def resnet50():
    """Constructs a ResNet-50 model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    return model

def preActResNet50():
    model = PreActResNet(Bottleneck, [3, 4, 6 ,3])
    return model
 
def resnet101():
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    return model
 
 
def resnet152():
    """Constructs a ResNet-152 model."""
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    return model

'''model = preActResNet50()
print(model.modules)

input  = torch.randn(1, 3, 224, 224)
print(input)
out = model(input)
classes = torch.max(out, 1) 
print(out)
print(classes)'''