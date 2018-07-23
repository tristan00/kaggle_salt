import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
import os
from PIL import Image
import glob
from distutils.version import LooseVersion

train_files_loc = '/home/td/Documents/salt/train/'


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out += shortcut
        out = self.relu(out)

        return out

class DeconvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2, stride=1, upsample=None):
        super(DeconvBottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if stride == 1:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=stride, bias=False, padding=1)
        else:
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels,
                                            kernel_size=3,
                                            stride=stride, bias=False,
                                            padding=1,
                                            output_padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.upsample = upsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.upsample is not None:
            shortcut = self.upsample(x)

        out += shortcut
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, downblock, upblock, num_layers, n_classes):
        super(ResNet, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.dlayer1 = self._make_downlayer(downblock, 64, num_layers[0])
        self.dlayer2 = self._make_downlayer(downblock, 128, num_layers[1],
                                            stride=2)
        self.dlayer3 = self._make_downlayer(downblock, 256, num_layers[2],
                                            stride=2)
        self.dlayer4 = self._make_downlayer(downblock, 512, num_layers[3],
                                            stride=2)

        self.uplayer1 = self._make_up_block(upblock, 512, 1, stride=2)
        self.uplayer2 = self._make_up_block(upblock, 256, num_layers[2], stride=2)
        self.uplayer3 = self._make_up_block(upblock, 128, num_layers[1], stride=2)
        self.uplayer4 = self._make_up_block(upblock, 64, 2, stride=2)

        upsample = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels,  # 256
                               64,
                               kernel_size=1, stride=2,
                               bias=False, output_padding=1),
            nn.BatchNorm2d(64),
        )
        self.uplayer_top = DeconvBottleneck(self.in_channels, 64, 1, 2, upsample)

        self.conv1_1 = nn.ConvTranspose2d(64, n_classes, kernel_size=1, stride=1,
                                 bias=False)

    def _make_downlayer(self, block, init_channels, num_layer, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != init_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, init_channels*block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(init_channels*block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, init_channels, stride, downsample))
        self.in_channels = init_channels * block.expansion
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels))

        return nn.Sequential(*layers)

    def _make_up_block(self, block, init_channels, num_layer, stride=1):
        upsample = None
        # expansion = block.expansion
        if stride != 1 or self.in_channels != init_channels * 2:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.in_channels, init_channels*2,
                                   kernel_size=1, stride=stride,
                                   bias=False, output_padding=1),
                nn.BatchNorm2d(init_channels*2),
            )
        layers = []
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels, 4))
        layers.append(block(self.in_channels, init_channels, 2, stride, upsample))
        self.in_channels = init_channels * 2
        return nn.Sequential(*layers)

    def forward(self, x):
        img = x
        x_size = x.size()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.dlayer1(x)
        x = self.dlayer2(x)
        x = self.dlayer3(x)
        x = self.dlayer4(x)

        x = self.uplayer1(x)
        x = self.uplayer2(x)
        x = self.uplayer3(x)
        x = self.uplayer4(x)
        x = self.uplayer_top(x)

        x = self.conv1_1(x, output_size=img.size())

        return x


def ResNet50(**kwargs):
    return ResNet(Bottleneck, DeconvBottleneck, [3, 4, 6, 3], 22, **kwargs)


def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 2], 22, **kwargs)


def generate_input_image_and_masks(batch_size = 32):
    mask_locs = list(glob.glob(train_files_loc + '/masks/*.png'))
    x, y = [], []

    for mask_loc in mask_locs:
        transpose = random.randint(0, 1)
        rotation = random.randint(1, 4)

        mask_name = os.path.basename(mask_loc).split('.')[0]
        image_loc = train_files_loc + '/images/{0}.png'.format(mask_name)
        start_image = Image.open(image_loc).convert('LA')
        np_image = np.array(start_image.getdata())[:, 0]
        np_image = np_image.reshape(start_image.size[1], start_image.size[0])

        start_mask = Image.open(mask_loc).convert('LA')
        np_mask = np.array(start_mask.getdata())[:, 0]
        np_mask = np_mask.reshape(start_mask.size[1], start_mask.size[0])

        np_mask = np_mask // 255

        if transpose == 1:
            np_image = np.transpose(np_image)
            np_mask = np.transpose(np_mask)

        np_image = np.rot90(np_image, rotation)
        np_mask = np.rot90(np_mask, rotation)

        np_image = np.pad(np_image, ((0, 27), (0, 27)), 'constant')
        np_mask = np.pad(np_mask, ((0, 27), (0, 27)), 'constant')
        #
        np_image = np.expand_dims(np_image, 0)
        np_mask = np.expand_dims(np_mask, 0)

        x.append(np_image)
        y.append(np_mask)
        if len(x) == batch_size:
            yield np.array(x), np.array(y)


def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(input)
    else:
        # >=0.3
        log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss


def train(model, device, optimizer):
    model.train()

    gen = generate_input_image_and_masks()

    for x, y in gen:
        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)
        output = model(x_t)
        loss = cross_entropy2d(output, y_t)
        loss.backward()
        optimizer.step()

        print(loss.item())



if __name__ == '__main__':
    device = torch.device('cuda')
    model = ResNet50().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train(model, device, optimizer)











