import torch
from torch import optim
import torchvision
import torch.nn as nn
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
	def __init__(self, stab_iter):
		super(Generator, self).__init__()
		self.stab_iter = stab_iter

		# adding first block
		first_block = []
		first_block += conv(512, 512, 4, 1, 3)
		first_block += conv(512, 512, 3, 1, 1)
		first_block = nn.Sequential(*first_block)

		# conv block sequential
		self.current_conv = nn.Sequential()
		self.current_conv.add_module('{}_block'.format(0), first_block)

		# torgb sequential
		self.to_rgb = nn.Sequential()

	def create_conv(self, x, flag_half=False):
		new_conv = []
		if flag_half:
			new_conv += conv(x.shape[1], int(x.shape[1]/2), 3, 1, 1, size_up='up')
			new_conv += conv(int(x.shape[1]/2), int(x.shape[1]/2), 3, 1, 1)
		else:
			new_conv += conv(x.shape[1], x.shape[1], 3, 1, 1, size_up='up')
			new_conv += conv(x.shape[1], x.shape[1], 3, 1, 1)
		return new_conv

	def add_new_layer(self, x, rindex, flag_half=False):
		x = self.current_conv(x)
		new_conv = self.create_conv(x, flag_half)
		new_conv = nn.Sequential(*new_conv).to(device)
		self.current_conv.add_module('{}_block'.format(rindex), new_conv)
		x = new_conv(x)

		to_rgb_layers = toRGB(x.shape[1])
		to_rgb_layers = nn.Sequential(*to_rgb_layers).to(device)
		self.to_rgb.add_module('{}_rgb'.format(rindex), to_rgb_layers)
		output = to_rgb_layers(x)
		return output

	def forward_first_block(self, x, rindex, flag_add):
		# forward to conv layers
		x = self.current_conv(x)

		# adding rgb layers to network after first growth
		if flag_add:
			to_rgb_layers = toRGB(x.shape[1])
			to_rgb_layers = nn.Sequential(*to_rgb_layers).to(device)
			self.to_rgb.add_module('{}_rgb'.format(rindex), to_rgb_layers)

		# forward to toRGB layers
		output = self.to_rgb(x)
		return output

	def grow_stab(self, x, rindex, flag_add, alpha_compo, flag_half=False):
		if flag_add:
			self.add_new_layer(x, rindex, flag_half)

		# grow network with variant alpha / stabilize with alpha=1

		model_keys = list(dict(self.current_conv.named_children()).keys())
		torgb_keys = list(dict(self.to_rgb.named_children()).keys())

		for key in model_keys[:-1]:
			x = dict(self.current_conv.named_children())[key](x)

		prev_conv = nn.Upsample(scale_factor=2, mode='nearest')(x)
		new_conv = dict(self.current_conv.named_children())[model_keys[-1]](x)

		prev_image = dict(self.to_rgb.named_children())[torgb_keys[-2]](prev_conv)
		new_image = dict(self.to_rgb.named_children())[torgb_keys[-1]](new_conv)
		output = (1 - alpha_compo) * prev_image + alpha_compo * new_image
		return output

	def forward(self, x, rindex, flag_add=False, flag_grow=False, alpha_compo=1):
		if rindex == 0:
			output = self.forward_first_block(x, rindex, flag_add)

		elif rindex >= 1 and rindex <=3:
			output = self.grow_stab(x, rindex, flag_add, alpha_compo)

		elif rindex >=4:
			output = self.grow_stab(x, rindex, flag_add, alpha_compo, flag_half=True)

		# output = nn.Tanh()(output)
		return output


class Discriminator(nn.Module):
	def __init__(self, ndf):
		super(Discriminator, self).__init__()
		# adding last block
		self.ndf = ndf
		last_block = []
		# add minibatch stddev later
		last_block += [Minibatch_std()]
		last_block += conv(self.ndf+1, self.ndf, 3, 1, 1, norm=None)
		last_block += conv(self.ndf, self.ndf, 4, 1, 0, norm=None)
		last_block = nn.Sequential(*last_block)

		# conv block sequential
		self.current_conv = nn.ModuleList()
		self.current_conv.append(last_block)

		# last linear layer
		self.linear = nn.Linear(ndf, 1)

		# fromrgb sequential
		self.from_rgb = nn.Sequential()


	def create_conv(self, x, flag_double=False):
		new_conv = []
		if flag_double:
			new_conv += conv(x.shape[1], 2*x.shape[1], 3, 1, 1)
			new_conv += conv(2*x.shape[1], 2*x.shape[1], 3, 1, 1, size_up='down')
		else:
			new_conv += conv(x.shape[1], x.shape[1], 3, 1, 1)
			new_conv += conv(x.shape[1], x.shape[1], 3, 1, 1, size_up='down')
		return new_conv

	def add_new_layer(self, x, rindex, flag_double=False):

		# create from_rgb
		if flag_double:
			denom = pow(2, rindex-3)
			from_rgb_layers = fromRGB(int(self.ndf/denom))
		else:
			from_rgb_layers = fromRGB(self.ndf)

		from_rgb_layers = nn.Sequential(*from_rgb_layers).to(device)
		self.from_rgb.add_module('{}_rgb'.format(9-rindex), from_rgb_layers)
		x = from_rgb_layers(x)

		# create new conv
		new_conv = self.create_conv(x, flag_double)
		new_conv = nn.Sequential(*new_conv).to(device)
		self.current_conv.insert(0, new_conv)
		# x = new_conv(x)

		# forward to conv layers
		for layer in self.current_conv:
			x = layer(x)
		output = x

		return output


	def grow_stab(self, x, rindex, flag_add, alpha_compo, flag_double=False):
		if flag_add:
			self.add_new_layer(x, rindex, flag_double)

		# grow network with variant alpha / stabilize with alpha=1
		model_keys = list(dict(self.current_conv.named_children()).keys())

		prev_rindex = rindex-1
		prev_from_rgb = dict(self.from_rgb.named_children())['{}_rgb'.format(9-prev_rindex)]
		prev_conv = list(dict(self.current_conv.named_children()).values())[1:]
		prev_image = nn.AvgPool2d(2, stride=2)(x)
		prev_feature = prev_from_rgb(prev_image)
		for layer in prev_conv:
			prev_feature = layer(prev_feature)

		new_rgb = dict(self.from_rgb.named_children())['{}_rgb'.format(9-rindex)]
		x = new_rgb(x)
		for layer in self.current_conv:
			x = layer(x)
		new_feature = x

		output = (1 - alpha_compo) * prev_feature + alpha_compo * new_feature
		return output

	def forward_last_block(self, x, rindex, flag_add):
		# adding rgb layers to network after first growth
		if flag_add:
			from_rgb_layers = fromRGB(c_out=self.ndf)
			from_rgb_layers = nn.Sequential(*from_rgb_layers).to(device)
			self.from_rgb.add_module('{}_rgb'.format(9-rindex), from_rgb_layers)

		# forward to toRGB layers
		x = self.from_rgb(x)

		# forward to conv layers
		for layer in self.current_conv:
			x = layer(x)

		return x



	def forward(self, x, rindex, flag_add=False, flag_grow=False, alpha_compo=1):
		if rindex == 0:
			output = self.forward_last_block(x, rindex, flag_add)

		elif rindex >= 1 and rindex <=3:
			output = self.grow_stab(x, rindex, flag_add, alpha_compo)

		elif rindex >=4:
			output = self.grow_stab(x, rindex, flag_add, alpha_compo, flag_double=True)

		# forward to last linear layer
		x = output.view(output.shape[0], -1)
		x = self.linear(x)

		output = nn.Sigmoid()(x)
		return output


if __name__ == '__main__':
	generator = Generator()
	random_tensor = torch.randn(4, 512, 4, 4)