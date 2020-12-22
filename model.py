import torch
from torch import optim
import torchvision
import torch.nn as nn
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
	def __init__(self, stab_iter, ngf):
		super(Generator, self).__init__()
		self.stab_iter = stab_iter
		self.ngf = ngf

		# adding first block
		first_block = []
		first_block += conv(self.ngf, self.ngf, 4, 1, 3)
		first_block += conv(self.ngf, self.ngf, 3, 1, 1)
		first_block = nn.Sequential(*first_block)

		# adding first rgb
		to_rgb_layers = toRGB(self.ngf)
		to_rgb_layers = nn.Sequential(*to_rgb_layers).to(device)

		# conv block sequential
		self.conv_layers = nn.Sequential()
		self.conv_layers.add_module('{}_block'.format(0), first_block)

		# torgb sequential
		self.to_rgb = nn.Sequential()
		self.to_rgb.add_module('{}_rgb'.format(0), to_rgb_layers)

	def add_new_layer(self, generator, x, rindex, flag_half=False):
		# forward to existing conv layers
		x = self.conv_layers(x)

		# make new conv layers
		new_conv = []
		if flag_half:
			new_conv += conv(x.shape[1], int(x.shape[1]/2), 3, 1, 1, size_up='up')
			new_conv += conv(int(x.shape[1]/2), int(x.shape[1]/2), 3, 1, 1)
		else:
			new_conv += conv(x.shape[1], x.shape[1], 3, 1, 1, size_up='up')
			new_conv += conv(x.shape[1], x.shape[1], 3, 1, 1)
		new_conv = nn.Sequential(*new_conv).to(device)

		# add new conv layers
		generator.module.conv_layers.add_module('{}_block'.format(rindex), new_conv)
		x = self.conv_layers[-1](x)

		# make new to_rgb layers
		to_rgb_layers = toRGB(x.shape[1])
		to_rgb_layers = nn.Sequential(*to_rgb_layers).to(device)

		# add new to_rgb layers
		generator.module.to_rgb.add_module('{}_rgb'.format(rindex), to_rgb_layers)
		output = self.to_rgb[-1](x)
		return output

	def grow_stab(self, x, rindex, alpha_compo, flag_half=False):
		# forward until last two conv layers
		for layer in self.conv_layers[:-1]:
			x = layer(x)

		# previous & last conv layers
		prev_conv = nn.Upsample(scale_factor=2, mode='nearest')(x)
		new_conv = self.conv_layers[-1](x)

		# forward to previous & last to_rgb layers
		prev_image = self.to_rgb[-2](prev_conv)
		new_image = self.to_rgb[-1](new_conv)

		# alpha composition
		output = (1 - alpha_compo) * prev_image + alpha_compo * new_image
		return output

	def forward(self, x, rindex, alpha_compo=1):
		if rindex == 0:
			x = self.conv_layers(x)
			output = self.to_rgb(x)

		elif rindex >= 1 and rindex <=3:
			output = self.grow_stab(x, rindex, alpha_compo)

		elif rindex >=4:
			output = self.grow_stab(x, rindex, alpha_compo, flag_half=True)

		output = nn.Tanh()(output)
		return output


class Discriminator(nn.Module):
	def __init__(self, ndf):
		super(Discriminator, self).__init__()
		self.ndf = ndf

		# adding last conv block
		last_block = []
		last_block += [Minibatch_std()]
		last_block += conv(self.ndf+1, self.ndf, 3, 1, 1, norm='spectral')
		last_block += conv(self.ndf, self.ndf, 4, 1, 0, norm='spectral')
		last_block = nn.Sequential(*last_block)

		# conv block sequential
		self.conv_layers = nn.ModuleList()
		self.conv_layers.append(last_block)

		# fromrgb sequential
		self.from_rgb = nn.Sequential()
		from_rgb_layers = fromRGB(c_out=self.ndf)
		from_rgb_layers = nn.Sequential(*from_rgb_layers).to(device)

		self.from_rgb.add_module('{}_rgb'.format(9), from_rgb_layers)

		# last linear layer
		self.linear = nn.Linear(ndf, 1)

	def add_new_layer(self, discriminator, x, rindex, flag_double=False):
		x = nn.Upsample(scale_factor=2, mode='nearest')(x)
		# make new from_rgb layers
		if flag_double:
			denom = pow(2, rindex-3)
			from_rgb_layers = fromRGB(int(self.ndf/denom))
		else:
			from_rgb_layers = fromRGB(self.ndf)

		# add new from_rgb layers
		from_rgb_layers = nn.Sequential(*from_rgb_layers).to(device)
		discriminator.module.from_rgb.add_module('{}_rgb'.format(9-rindex), from_rgb_layers)

		# forward to new from_rgb layers
		x = self.from_rgb[-1](x)

		# make new conv layers
		new_conv = []
		if flag_double:
			new_conv += conv(x.shape[1], 2*x.shape[1], 3, 1, 1, norm='spectral')
			new_conv += conv(2*x.shape[1], 2*x.shape[1], 3, 1, 1, norm='spectral', size_up='down')
		else:
			new_conv += conv(x.shape[1], x.shape[1], 3, 1, 1,norm='spectral')
			new_conv += conv(x.shape[1], x.shape[1], 3, 1, 1,norm='spectral', size_up='down')
		new_conv = nn.Sequential(*new_conv).to(device)

		# add new conv layers
		discriminator.module.conv_layers.insert(0, new_conv)

		# forward to conv layers
		for layer in self.conv_layers:
			x = layer(x)

		return x


	def grow_stab(self, x, rindex, alpha_compo, flag_double=False):
		# get from_rgb parameters and names
		fromrgb_dict = dict(self.from_rgb.named_children())

		prev_rindex = rindex-1

		# previous fromrgb & conv
		prev_from_rgb = fromrgb_dict['{}_rgb'.format(9-prev_rindex)]
		prev_conv = self.conv_layers[1:]

		# forward to previous fromrgb
		prev_image = nn.AvgPool2d(2, stride=2)(x)
		prev_feature = prev_from_rgb(prev_image)

		# forward until previous conv layers
		for layer in prev_conv:
			prev_feature = layer(prev_feature)

		# new fromrgb
		new_rgb = fromrgb_dict['{}_rgb'.format(9-rindex)]
		new_feature = new_rgb(x)

		# forward to new fromrgb
		for layer in self.conv_layers:
			new_feature = layer(new_feature)

		# alpha composition
		output = (1 - alpha_compo) * prev_feature + alpha_compo * new_feature
		return output

	def forward(self, x, rindex, alpha_compo=1):
		if rindex == 0:
			x = self.from_rgb(x)
			for layer in self.conv_layers:
				x = layer(x)

		elif rindex >= 1 and rindex <=3:
			x = self.grow_stab(x, rindex, alpha_compo)

		elif rindex >=4:
			x = self.grow_stab(x, rindex, alpha_compo, flag_double=True)

		# forward to last linear layer
		x = x.view(x.shape[0], -1)
		output = self.linear(x)
		output = nn.Sigmoid()(output)
		return output


if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	rindex = 0
	generator = Generator(4, 512).to(device)
	discriminator = Discriminator(512).to(device)

	generator = nn.DataParallel(generator)
	discriminator = nn.DataParallel(discriminator)

	batch_size = 4
	real_image = torch.randn(batch_size, 3, 4, 4).to(device)
	latent_vector = torch.randn(batch_size, 512, 1, 1).to(device)
	real_label, fake_label = torch.ones(batch_size).to(device), torch.zeros(batch_size).to(device)

	rindex += 1
	generator.module.add_new_layer(generator, latent_vector, rindex)
	discriminator.module.add_new_layer(discriminator, real_image, rindex)

	print('generator', generator)
	print('discriminator', discriminator)

	import pdb; pdb.set_trace()
