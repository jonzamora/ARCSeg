'''
SegNet
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

class encoder(nn.Module):
    '''
        Encoder for the Segmentation network
    '''

    def __init__(self, batchNorm_momentum, num_classes=23):
        super(encoder, self).__init__()
        self.batchNorm_momentum = batchNorm_momentum
        self.num_classes = num_classes
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, dilation=1, bias=False),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 4, 2, 1, dilation=1, bias=False),
            nn.BatchNorm2d(128, momentum=batchNorm_momentum),
            nn.ReLU(True),

            nn.Conv2d(128, 256, 4, 2, 1, dilation=1, bias=False),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.ReLU(True),

            nn.Conv2d(256, 512, 4, 2, 1, dilation=1, bias=False),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.ReLU(True),

            nn.Conv2d(512, 1024, 4, 1, 0, dilation=1, bias=False),
            nn.BatchNorm2d(1024, momentum=batchNorm_momentum),
            nn.ReLU(True)
        )

    def forward(self, input):
        output = self.main(input)
        return output

class decoder(nn.Module):
    '''
        Decoder for the Segmentation Network
    '''

    def __init__(self, batchNorm_momentum, num_classes=23):
        super(decoder, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512, momentum=batchNorm_momentum),
            nn.Dropout2d(),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256, momentum=batchNorm_momentum),
            nn.Dropout2d(),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128, momentum=batchNorm_momentum),
            nn.Dropout2d(),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64, momentum=batchNorm_momentum),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, num_classes, 4, 2, 1, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        output = self.main(input)
        return output

class SegNet(nn.Module):
    '''
        Segnet network
    '''

    def __init__(self, batchNorm_momentum, num_classes=23):
        super(SegNet, self).__init__()
        self.batchNorm_momentum = batchNorm_momentum
        self.num_classes = num_classes
        self.encoder = encoder(self.batchNorm_momentum, self.num_classes)
        self.decoder = decoder(self.batchNorm_momentum, self.num_classes)

    def forward(self, x):
        latent = self.encoder(x)
        #print('Latent Shape')
        #print(latent.shape)
        output = self.decoder(latent)
        #print('Output Shape')
        #print(output.shape)

        return output

    def dice_loss(self, output, target, weights=None, ignore_index=None):
        '''
            output : NxCxHxW Variable
            target :  NxHxW LongTensor
            weights : C FloatTensor
            ignore_index : int index to ignore from loss
        '''
        eps = 0.0001

        encoded_target = output.detach() * 0
        #print("encoded target:", encoded_target.shape)
        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = 1

        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + encoded_target

        if ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = weights * (1 - (numerator / denominator))

        return loss_per_channel.sum() / output.size(1)
