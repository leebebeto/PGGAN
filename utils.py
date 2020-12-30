import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from torch.nn.init import kaiming_normal, calculate_gain

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


class equalized_conv2d(nn.Module):
    def __init__(self, c_in, c_out, k_size, stride, pad, initializer='kaiming', bias=False):
        super(equalized_conv2d, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False)
        kaiming_normal(self.conv.weight, a=calculate_gain('conv2d'))

        conv_w = self.conv.weight.data.clone()
        self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))
        self.scale = (torch.mean(self.conv.weight.data ** 2)) ** 0.5
        self.conv.weight.data.copy_(self.conv.weight.data / self.scale)

    def forward(self, x):
        x = self.conv(x.mul(self.scale))
        return x + self.bias.view(1, -1, 1, 1).expand_as(x)


class equalized_linear(nn.Module):
    def __init__(self, c_in, c_out, initializer='kaiming'):
        super(equalized_linear, self).__init__()
        self.linear = nn.Linear(c_in, c_out, bias=False)
        if initializer == 'kaiming':
            kaiming_normal(self.linear.weight, a=calculate_gain('linear'))
        elif initializer == 'xavier':
            torch.nn.init.xavier_normal(self.linear.weight)

        linear_w = self.linear.weight.data.clone()
        self.bias = torch.nn.Parameter(torch.FloatTensor(c_out).fill_(0))
        self.scale = (torch.mean(self.linear.weight.data ** 2)) ** 0.5
        self.linear.weight.data.copy_(self.linear.weight.data / self.scale)

    def forward(self, x):
        x = self.linear(x.mul(self.scale))
        return x + self.bias.view(1, -1).expand_as(x)

def conv(c_in, c_out, k_size=3, stride=1, pad=0, equal=True, act='leaky', norm='pixel', size_up=None, bias=False, pixel=True):
    layers = []

    # upsample if needed
    if size_up == 'up':
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))

    # normalization
    if norm == 'spectral':
        layers.append(spectral_norm(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False)))
    else:
        # add conv layer & leaky relu as activation function
        if equal:
            layers.append(equalized_conv2d(c_in, c_out, k_size, stride, pad, bias=False))
        else:
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gradient_penalty(y, x):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)