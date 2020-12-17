from config import arg_parse
from data import *
from model import *
from utils import *
from config import *

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image

# receive configuration
args = arg_parse()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data = Celeb_HQ(train= True, rindex=args.rindex)

generator = Generator(args.stab_iter).to(device)
discriminator = Discriminator(args.ndf).to(device)

# generator = nn.DataParallel(generator)
# discriminator = nn.DataParallel(discriminator)
#
g_optimizer = optim.Adam(generator.parameters(), lr = args.lr, betas = (0, 0.99))
d_optimizer = optim.Adam(discriminator.parameters(), lr = args.lr, betas = (0, 0.99))

# criterion = nn.BCEWithLogitsLoss()
criterion = nn.MSELoss()



step = 0
total_step = 9 * (2 * args.stab_iter)
print('total_step: {}'.format(total_step))

train_data.renew(initial=True)
while step < total_step:
	# alpha_compo = 0.0
	real_image, _ = next(iter(train_data.train_loader))
	real_image = real_image.to(device)
	# latent_vector = torch.FloatTensor(train_data.batch_size, args.z_dim).to(device)
	latent_vector = torch.rand(train_data.batch_size, args.z_dim).to(device)
	real_label, fake_label = torch.ones(train_data.batch_size, 1).to(device), torch.zeros(train_data.batch_size, 1).to(device)
	# save_image(real_image, f'real_image_{step}_{train_data.rindex}.png', normalize=True)
	# initialize for the first block
	if train_data.rindex == 0:
		flag_add = False

		# print(list(generator.parameters())[0])
		# import pdb;
		# pdb.set_trace()

		# adding network
		if step % (args.stab_iter*2) == 0:
			print('adding first block')
			fake_image = generator(latent_vector.unsqueeze(2).unsqueeze(2), train_data.rindex, flag_add=True)
			real_pred = discriminator(real_image, train_data.rindex, flag_add=True)
			fake_pred = discriminator(fake_image, train_data.rindex)

		else:
			fake_image = generator(latent_vector.unsqueeze(2).unsqueeze(2), train_data.rindex)
			real_pred = discriminator(real_image, train_data.rindex)
			fake_pred = discriminator(fake_image, train_data.rindex)

		# print(list(generator.parameters())[0])
		# import pdb;
		# pdb.set_trace()

	else:
		flag_grow = False
		# grow network
		if step >= (train_data.rindex) * (2*args.stab_iter) and step < (train_data.rindex) * (2*args.stab_iter) + args.stab_iter:
			# adding network
			alpha_compo = (step - (train_data.rindex * args.stab_iter * 2)) / (args.stab_iter)
			if step % (train_data.rindex * (2*args.stab_iter)) == 0:
				print('---------------- adding & growing network ----------------')
				fake_image = generator(latent_vector.unsqueeze(2).unsqueeze(2), train_data.rindex, flag_add=True, alpha_compo=0)
				real_pred = discriminator(real_image, train_data.rindex, flag_add=True, alpha_compo=0)
				fake_pred = discriminator(fake_image, train_data.rindex)
			else:
				fake_image = generator(latent_vector.unsqueeze(2).unsqueeze(2), train_data.rindex, flag_grow=True, alpha_compo= alpha_compo)
				real_pred = discriminator(real_image, train_data.rindex, flag_grow=True, alpha_compo= alpha_compo)
				fake_pred = discriminator(fake_image, train_data.rindex, flag_grow=True, alpha_compo= alpha_compo)
			print('alpha: {}, step: {}, rindex: {}, real_image: {}, fake_image: {} real_pred: {}, fake_pred: {}'.format(alpha_compo, step, train_data.rindex, real_image.shape, fake_image.shape, real_pred.shape, fake_pred.shape))
		# print(fake_image, alpha_compo)

		# stabilize network
		elif step >= (train_data.rindex) * (2*args.stab_iter) + args.stab_iter and step <= (train_data.rindex) * (2*args.stab_iter) + (2*args.stab_iter):
			alpha_compo = 1.0
			if step % ((train_data.rindex * (2*args.stab_iter)) + args.stab_iter) == 0:
				print('---------------- stabilize network ----------------')
			fake_image = generator(latent_vector.unsqueeze(2).unsqueeze(2), train_data.rindex, flag_grow=True, alpha_compo= alpha_compo)
			real_pred = discriminator(real_image, train_data.rindex, flag_grow=True, alpha_compo= alpha_compo)
			fake_pred = discriminator(fake_image, train_data.rindex, flag_grow=True, alpha_compo= alpha_compo)
			print('alpha: {}, step: {}, rindex: {}, real_image: {}, fake_image: {} real_pred: {}, fake_pred: {}'.format(alpha_compo, step, train_data.rindex, real_image.shape, fake_image.shape, real_pred.shape, fake_pred.shape))
			# print(alpha_compo)

	# print('real_pred', real_pred)
	# print('fake_pred', fake_pred)
	# print(fake_image)
	# save_image(fake_image, f'fake_image_{train_data.rindex}_{step}.png', normalize=True)
	# import pdb; pdb.set_trace()
	#
	# # # print('alpha_compo', alpha_compo)
	# #
	# # # update discriminator
	d_real_loss = criterion(real_pred, real_label)
	d_fake_loss = criterion(fake_pred.detach(), fake_label)

	if step % 10 == 0:
		print('real_pred', real_pred)
		print('fake_pred', fake_pred)
		save_image(fake_image, 'fake_image_{}.png'.format(step), normalize=True)
		# print('growing, step: {}, alpha_compo: {}, rindex: {}, real_image: {}, fake_image: {} real_pred: {}, fake_pred: {}'.format(step, alpha_compo, train_data.rindex, real_image.shape, fake_image.shape, real_pred.shape, fake_pred.shape))
		print('growing, step: {}, rindex: {}, real_image: {}, fake_image: {} real_pred: {}, fake_pred: {}'.format(step, train_data.rindex, real_image.shape, fake_image.shape, real_pred.shape, fake_pred.shape))

	# d_loss = 0.5 * (d_real_loss + d_fake_loss)
	d_loss = d_real_loss + d_fake_loss
	d_optimizer.zero_grad()
	d_loss.backward(retain_graph=True)
	d_optimizer.step()

	# fake_pred_new = fake_pred.detach().clone()
	# fake_pred_new.requires_grad = True
	alpha_compo=0.0
	fake_pred = discriminator(fake_image, train_data.rindex, flag_grow=True, alpha_compo=alpha_compo)
	g_loss = criterion(fake_pred, real_label)
	g_optimizer.zero_grad()
	g_loss.backward()
	g_optimizer.step()
	#
	print(real_image)
	save_image(real_image, f'real_image_{step}.png', normalize=True)
	print(fake_image)
	save_image(fake_image, f'fake_image_{step}.png', normalize=True)
	#
	print('d_loss: {}, g_loss: {}'.format(d_loss, g_loss))
	# import pdb; pdb.set_trace()
	# # growing the image resolution
	#
	# # print('generator', generator)
	# # print('discriminator', discriminator)

	if step > 0 and (step+1) % (args.stab_iter*2) == 0:
		print('Increasing resolution index: ', train_data.rindex)


		if train_data.rindex <8:
			train_data.renew()
		print('Increased resolution index: ', train_data.rindex)

	step += 1


# for epoch in range(args.num_epochs):
# 	for i, (images) in enumerate(train_loader):




