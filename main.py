import os

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

generator = Generator(args.stab_iter, args.ngf).to(device)
discriminator = Discriminator(args.ndf).to(device)

generator = nn.DataParallel(generator)
discriminator = nn.DataParallel(discriminator)

g_optimizer = optim.Adam(generator.parameters(), lr = args.lr, betas = (0, 0.99))
d_optimizer = optim.Adam(discriminator.parameters(), lr = args.lr, betas = (0, 0.99))

# criterion = nn.MSELoss()
criterion = nn.BCELoss()

# making logging folders
os.makedirs('random_images', exist_ok=True)
os.makedirs('fixed_images', exist_ok=True)
os.makedirs('model', exist_ok=True)


step = 0
total_step = 9 * (2 * args.stab_iter)
train_data.renew(initial=True)
fixed_vector = torch.FloatTensor(train_data.batch_size, args.z_dim, 1, 1).to(device)

if args.retrain:
	generator.load_state_dict(torch.load( f'model/generator_{args.retrain_resol}*{args.retrain_resol}_latest.pt'))
	discriminator.load_state_dict(torch.load( f'model/discriminator_{args.retrain_resol}*{args.retrain_resol}_latest.pt'))

print(f'Start training PGGAN with total_step of {total_step}')
while step < total_step:

	# clear accumulated gradients
	d_optimizer.zero_grad()
	g_optimizer.zero_grad()

	# steps for grow and stabilize
	grow_start = train_data.rindex * (2 * args.stab_iter)
	grow_end = grow_start + args.stab_iter
	stab_end = grow_end + args.stab_iter

	real_image, _ = next(iter(train_data.train_loader))
	real_image = real_image.to(device)
	latent_vector = torch.randn(train_data.batch_size, args.z_dim, 1, 1).to(device)
	real_label, fake_label = torch.ones(train_data.batch_size).to(device), torch.zeros(train_data.batch_size).to(device)

	# flags for adding network -> add only if step == grow_start
	flag_add= False
	if grow_start>0 and step % grow_start ==0: flag_add = True

	# alpha compo with grow/stabilize stage
	alpha_compo = (step - (grow_start))/args.stab_iter
	if step >= grow_end and step<=stab_end : alpha_compo = 1.0

	##################################################
	############# UPDATE DISCRIMINATOR ###############
	##################################################

	# grow network -> grow only at first
	fake_image = generator(latent_vector, train_data.rindex, alpha_compo=alpha_compo)
	real_pred = discriminator(real_image, train_data.rindex, alpha_compo=alpha_compo).view(-1)
	fake_pred = discriminator(fake_image.detach(), train_data.rindex).view(-1)

	d_real_loss = criterion(real_pred, real_label)
	d_fake_loss = criterion(fake_pred, fake_label)

	d_loss = d_real_loss + d_fake_loss
	d_loss.backward()
	d_optimizer.step()

	##################################################
	############# UPDATE GENERATOR ###################
	##################################################

	latent_vector = torch.randn(train_data.batch_size, args.z_dim, 1, 1).to(device)
	fake_image = generator(latent_vector, train_data.rindex, alpha_compo= alpha_compo)
	fake_pred = discriminator(fake_image, train_data.rindex, alpha_compo=alpha_compo).view(-1)

	g_loss = criterion(fake_pred, real_label)
	g_loss.backward()
	g_optimizer.step()

	##################################################
	################### LOGGING ######################
	##################################################
	if step % args.print_freq == 0:
		print(f'growing, step: {step}, rindex: {train_data.rindex}, d_loss: {d_loss}, g_loss: {g_loss}')

	if step % args.save_image_freq == 0:
		#saving random images
		resol = pow(2, train_data.rindex+2)
		save_image(fake_image, f'random_images/{resol}*{resol}_{step}.png', normalize=True)

		# saving fixed images
		fake_image = generator(fixed_vector, train_data.rindex, alpha_compo= alpha_compo)
		save_image(fake_image, f'fixed_images/{resol}*{resol}_{step}.png', normalize=True)

	if step % args.save_model_freq == 0:
		# saving model with steps
		torch.save(generator.state_dict(), f'model/{resol}*{resol}_{step}.pt')
		torch.save(discriminator.state_dict(), f'model/{resol}*{resol}_{step}.pt')

	##################################################
	############# INCREASING IMAGE RESOLUTION ########
	##################################################

	if step > 0 and (step+1) % (args.stab_iter*2) == 0:
		print(f'Increasing resolution index: {train_data.rindex} -> {train_data.rindex+1}')
		if train_data.rindex <8:
			# saving latest model of the resolution
			torch.save(generator.state_dict(), f'model/generator_{resol}*{resol}_latest.pt')
			torch.save(discriminator.state_dict(), f'model/discriminator_{resol}*{resol}_latest.pt')

			# increasing resolution
			train_data.renew()
			generator.module.add_new_layer(generator, latent_vector, train_data.rindex)
			discriminator.module.add_new_layer(discriminator, real_image, train_data.rindex)

	step += 1




