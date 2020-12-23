from utils import *

import torch
import torch.nn as nn
from torch import optim
import torchvision
from torchvision.utils import save_image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
	def __init__(self, stab_iter, ngf):
		super(Generator, self).__init__()
		self.stab_iter = stab_iter
		self.ngf = ngf

		# channel dictionary
		self.rindex2ch = [self.ngf,self.ngf,self.ngf,self.ngf,
						  self.ngf,int(self.ngf/2),int(self.ngf/4),int(self.ngf/8),int(self.ngf/16)]

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

	def add_new_layer(self, rindex, flag_half=False):
		ch = self.rindex2ch[rindex]

		# make new conv layers
		new_conv = []
		if flag_half:
			new_conv += conv(ch, int(ch/2), 3, 1, 1, size_up='up')
			new_conv += conv(int(ch/2), int(ch/2), 3, 1, 1)
			rgb_ch = int(ch/2)
		else:
			new_conv += conv(ch, ch, 3, 1, 1, size_up='up')
			new_conv += conv(ch, ch, 3, 1, 1)
			rgb_ch = ch

		new_conv = nn.Sequential(*new_conv).to(device)

		# add new conv layers
		self.conv_layers.add_module('{}_block'.format(rindex), new_conv)

		# make new to_rgb layers
		to_rgb_layers = toRGB(rgb_ch)
		to_rgb_layers = nn.Sequential(*to_rgb_layers).to(device)

		# add new to_rgb layers
		self.to_rgb.add_module('{}_rgb'.format(rindex), to_rgb_layers)

	def grow_stab(self, x, rindex, alpha_compo):
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

		else:
			output = self.grow_stab(x, rindex, alpha_compo)

		output = nn.Tanh()(output)
		return output


class Discriminator(nn.Module):
	def __init__(self, ndf):
		super(Discriminator, self).__init__()
		self.ndf = ndf

		# channel dictionary
		self.rindex2ch = [self.ndf,self.ndf,self.ndf,self.ndf,
						  self.ndf,int(self.ndf/2),int(self.ndf/4), int(self.ndf/8),int(self.ndf/16)]


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

	def add_new_layer(self, rindex, flag_double=False):
		ch = self.rindex2ch[rindex]

		# make new from_rgb layers
		if flag_double:
			denom = pow(2, rindex-3)
			from_rgb_layers = fromRGB(int(self.ndf/denom))
		else:
			from_rgb_layers = fromRGB(self.ndf)

		# add new from_rgb layers
		from_rgb_layers = nn.Sequential(*from_rgb_layers).to(device)
		self.from_rgb.add_module('{}_rgb'.format(9-rindex), from_rgb_layers)

		# make new conv layers
		new_conv = []
		if flag_double:
			new_conv += conv(int(ch/2), ch, 3, 1, 1, norm='spectral')
			new_conv += conv(ch, ch, 3, 1, 1, norm='spectral', size_up='down')
		else:
			new_conv += conv(ch, ch, 3, 1, 1,norm='spectral')
			new_conv += conv(ch, ch, 3, 1, 1,norm='spectral', size_up='down')
		new_conv = nn.Sequential(*new_conv).to(device)

		# add new conv layers
		self.conv_layers.insert(0, new_conv)

	def grow_stab(self, x, rindex, alpha_compo):


		# get from_rgb parameters and names
		fromrgb_dict = dict(self.from_rgb.named_children())

		prev_rindex = rindex-1

		# previous fromrgb
		prev_from_rgb = fromrgb_dict['{}_rgb'.format(9-prev_rindex)]
		prev_image = nn.AvgPool2d(2, stride=2)(x)
		prev_feature = prev_from_rgb(prev_image)

		# new fromrgb
		new_rgb = fromrgb_dict['{}_rgb'.format(9-rindex)]
		new_feature = new_rgb(x)
		new_feature = self.conv_layers[0](new_feature)

		output = (1 - alpha_compo) * prev_feature + alpha_compo * new_feature

		# forward until previous conv layers
		for layer in self.conv_layers[1:]:
			output = layer(output)
			# if rindex == 5:
			# 	print(output)
			# 	import pdb; pdb.set_trace()
			# 	print()

		return output

	def forward(self, x, rindex, alpha_compo=1):
		if rindex == 0:
			x = self.from_rgb(x)
			for layer in self.conv_layers:
				x = layer(x)
		else:
			x = self.grow_stab(x, rindex, alpha_compo)

		# forward to last linear layer
		x = x.view(x.shape[0], -1)
		output = self.linear(x)
		output = nn.Sigmoid()(output)
		return output


if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	rindex = 0
	rindex2batch = {0: 16, 1: 16, 2: 16, 3: 16, 4: 16, 5: 4, 6: 4, 7: 1, 8: 1}

	generator = Generator(4, 512).to(device)
	discriminator = Discriminator(512).to(device)

	generator = nn.DataParallel(generator)
	discriminator = nn.DataParallel(discriminator)

	for i in range(8):
		resol = pow(2, rindex+2)

		batch_size = rindex2batch[rindex]
		real_image = torch.randn(batch_size, 3, resol, resol).to(device)
		latent_vector = torch.randn(batch_size, 512, 1, 1).to(device)
		real_label, fake_label = torch.ones(batch_size).to(device), torch.zeros(batch_size).to(device)

		rindex += 1
		generator.module.add_new_layer(generator, latent_vector, rindex)
		discriminator.module.add_new_layer(discriminator, real_image, rindex)
