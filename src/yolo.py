from __future__ import print_function
import torch

from torch import nn
from torch.autograd import Variable


class CNNWithMaxPoolingReLU(nn.Module):
    def __init__(self, **kwargs):
        super(CNNWithMaxPoolingReLU, self).__init__()

        in_channels = self.in_channels = kwargs.get('in_channels', 3)
        out_channels = self.out_channels = kwargs.get('out_channels', 5)
        kernel_size = self.kernel_size = kwargs.get('kernel_size', 5)
        stride = self.stride = kwargs.get('stride', 2)
        pool_size = self.pool_size = kwargs.get('pool_size', 2)

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.ReLU(),
            nn.MaxPool2d(pool_size),
        )

    def forward(self, x):
        return self.cnn(x)

    def __repr__(self):
        return 'CNNWithBatchNormalReLU(kernel_size = %s inChannles= %i, oChannels= %i)' % (
            str(self.kernel_size), self.in_channels, self.out_channels)


theYoloConfigs = [
    {"in_channels": 3, "out_channels": 10, "kernel_size": 2, "stride": 1},
    {"in_channels": 10, "out_channels": 20, "kernel_size": 4, "stride": 1},
    {"in_channels": 20, "out_channels": 40, "kernel_size": 4, "stride": 1},
    {"in_channels": 40, "out_channels": 50, "kernel_size": 16, "stride": 1},
    {"in_channels": 50, "out_channels": 50, "kernel_size": (10, 5), "stride": 1},
]


class YoLo(nn.Module):
    def __init__(self, configs):
        super(YoLo, self).__init__()

        cnns = map(lambda config: CNNWithMaxPoolingReLU(**config), configs)

        self.convs = nn.Sequential(
            *cnns,
            nn.Linear()
        )

    def forward(self, x):
        return self.convs.forward(x)


import time

if __name__ == '__main__':
    yolo = YoLo(theYoloConfigs)

    print(yolo)

    start = time.time()

    if torch.cuda.is_available():
        yolo = yolo.cuda()
        start = time.time()
        for _ in range(10):
            images = Variable(torch.rand(2, 3, 800, 600).cuda())
            prediction = yolo.forward(images)
    else:
        prediction = yolo.forward(Variable(torch.rand(1, 3, 800, 600)))
    print('used time', (time.time() - start) * 1000, 'ms')

    prediction = prediction.cpu()
    print(prediction.size()[1:])
