import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

class Pixel_norm(nn.Module):
    def __init__(self):
        super(Pixel_norm, self).__init__()
        self.eps= 1e-8

    def forward(self, x):
        x = x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.eps)
        return x

class Minibatch_std(nn.Module):
    def __init__(self):
        super(Minibatch_std, self).__init__()

    def forward(self, x):
        new_layer = torch.mean(torch.std(x, dim=0), dim=0)
        output = torch.cat((x, new_layer.expand(x.shape[0], 1, x.shape[2], x.shape[3])), dim=1)
        return output

def conv(c_in, c_out, k_size=3, stride=1, pad=0, act='leaky', norm='pixel', size_up=None, bias=False, pixel=True):
    layers = []

    # upsample if needed
    if size_up == 'up':
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))

    # normalization
    if norm == 'spectral':
        layers.append(spectral_norm(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False)))
    else:
        # add conv layer & leaky relu as activation function
        layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))

        # pixel norm
        if norm == 'pixel':
            layers.append(Pixel_norm())

    # activation function
    if act == 'leaky':
        layers.append(nn.LeakyReLU(0.2))
    elif act == 'none':
        pass

    # downsample if needed
    if size_up == 'down':
        layers.append(nn.AvgPool2d(2, stride=2))


    return layers

def toRGB(c_in):
    layers = conv(c_in, 3, k_size=1, stride=1, pad=0, act='none', norm=None)
    return layers

def fromRGB(c_out):
    layers = conv(3, c_out, k_size=1, stride=1, pad=0, act='leaky', norm=None)
    return layers
