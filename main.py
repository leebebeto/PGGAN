import os
import datetime
import time

from config import arg_parse
from data import *
from model import *
from utils import *
from config import *

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms
from torchvision.utils import save_image

torch.backends.cudnn.benchmark = True  # boost speed.

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

criterion = nn.MSELoss()
# criterion = nn.BCELoss()
# criterion = nn.DataParallel(criterion)

# making logging folders/tensorboard
os.makedirs('random_images', exist_ok=True)
os.makedirs('fixed_images', exist_ok=True)
os.makedirs('model', exist_ok=True)
writer = SummaryWriter()

step = 0
total_step = 9 * (2 * args.stab_iter)
train_data.renew(initial=True)
fixed_vector = torch.randn(train_data.batch_size, args.z_dim, 1, 1).to(device)

if args.retrain:
	step += args.stab_iter * 2
	print('loading pretraind network...')
	generator.load_state_dict(torch.load( f'model/generator_{args.retrain_resol}*{args.retrain_resol}_latest.pt'))
	discriminator.load_state_dict(torch.load( f'model/discriminator_{args.retrain_resol}*{args.retrain_resol}_latest.pt'))

	latent_vector = torch.randn(train_data.batch_size, args.z_dim, 1, 1).to(device)
	real_image, _ = next(iter(train_data.train_loader))
	real_image = real_image.to(device)

	# increasing resolution
	train_data.renew()
	generator.module.add_new_layer(generator, latent_vector, train_data.rindex)
	discriminator.module.add_new_layer(discriminator, real_image, train_data.rindex)

print(f'Start training PGGAN with total_step of {total_step}')
prev_time = time.time()
train_loader = iter(train_data.train_loader)
while step < total_step:

	# clear accumulated gradients
	d_optimizer.zero_grad()
	g_optimizer.zero_grad()

	# steps for grow and stabilize
	grow_start = train_data.rindex * (2 * args.stab_iter)
	grow_end = grow_start + args.stab_iter
	stab_end = grow_end + args.stab_iter

	try:
		real_image = next(train_loader)
	except:
		train_loader = iter(train_data.train_loader)
		real_image = next(train_loader)

	real_image = real_image.to(device)
	latent_vector = torch.randn(train_data.batch_size, args.z_dim, 1, 1).to(device)
	real_label, fake_label = torch.ones(train_data.batch_size).to(device), torch.zeros(train_data.batch_size).to(device)

	# alpha compo with grow/stabilize stage
	alpha_compo = (step - (grow_start))/args.stab_iter
	if step >= grow_end and step<=stab_end:
		alpha_compo = 1.0

	# ##################################################
	# ############# UPDATE DISCRIMINATOR ###############
	# ##################################################
	fake_image = generator(latent_vector, train_data.rindex, alpha_compo)
	real_pred = discriminator(real_image, train_data.rindex, alpha_compo).view(-1)
	fake_pred = discriminator(fake_image.detach(), train_data.rindex, alpha_compo).view(-1)

	d_real_loss = criterion(real_pred, real_label)
	d_fake_loss = criterion(fake_pred, fake_label)

	d_loss = d_real_loss + d_fake_loss
	d_loss.backward()
	d_optimizer.step()

	##################################################
	############# UPDATE GENERATOR ###################
	##################################################

	fake_image_new = generator(latent_vector, train_data.rindex, alpha_compo)
	fake_pred_new = discriminator(fake_image_new, train_data.rindex, alpha_compo).view(-1)

	g_loss = criterion(fake_pred_new, real_label)
	g_loss.backward()
	g_optimizer.step()

	##################################################
	################### LOGGING ######################
	##################################################
	step_left = total_step - step
	time_took = time.time() - prev_time
	time_left = datetime.timedelta(seconds=step_left * (time_took))
	prev_time = time.time()

	if step % args.print_freq == 0:
		print(f'step: {step}, alpha: {alpha_compo}, rindex: {train_data.rindex}, d_loss: {d_loss:.4f}, g_loss: {g_loss:.4f}, time took: {time_took:.4f}')
		print(f'time left: {time_left}')
		writer.add_scalar("d_loss/step", d_loss.item(), step)
		writer.add_scalar("g_loss/step", g_loss.item(), step)
		writer.add_scalar("time/step", time_took, step)
		writer.add_scalar("alpha_compo/step", alpha_compo, step)

	if step % args.save_image_freq == 0:
		#saving random images
		resol = pow(2, train_data.rindex+2)
		save_image(fake_image, f'random_images/{resol}*{resol}_{step}.png', normalize=True)

		# saving fixed images
		fake_image = generator(fixed_vector, train_data.rindex, alpha_compo= alpha_compo)
		save_image(fake_image, f'fixed_images/{resol}*{resol}_{step}.png', normalize=True)

	if step % args.save_model_freq == 0:
		# saving model with steps
		torch.save(generator.state_dict(), f'model/generator_{resol}*{resol}_{step}.pt')
		torch.save(discriminator.state_dict(), f'model/discriminator_{resol}*{resol}_{step}.pt')

	##################################################
	############# INCREASING IMAGE RESOLUTION ########
	##################################################
	flag_add = False
	if step > 0 and (step + 1) % (args.stab_iter*2) == 0:
		flag_add = True

	if flag_add:
		if train_data.rindex  == 8:
			break

		print(f'Increasing resolution index: {train_data.rindex} -> {train_data.rindex+1}')
		if train_data.rindex <8:
			# saving latest model of the resolution
			torch.save(generator.state_dict(), f'model/generator_{resol}*{resol}_latest.pt')
			torch.save(discriminator.state_dict(), f'model/discriminator_{resol}*{resol}_latest.pt')

			# increasing resolution
			train_data.renew()
			train_loader = iter(train_data.train_loader)
			flag_half = True if train_data.rindex >= 4 else False
			flag_double = True if train_data.rindex >= 4 else False

			generator.module.add_new_layer(train_data.rindex, flag_half=flag_half)
			discriminator.module.add_new_layer(train_data.rindex, flag_double=flag_double)


	step += 1






	#
	#
	# ##################################################
	# ############# UPDATE DISCRIMINATOR ###############
	# ##################################################
	# fake_image = generator(latent_vector, train_data.rindex, alpha_compo=alpha_compo)
	# real_pred = discriminator(real_image, train_data.rindex, alpha_compo=alpha_compo).view(-1)
	# fake_pred = discriminator(fake_image.detach(), train_data.rindex).view(-1)
	#
	# d_real_loss = -real_pred.mean()
	# d_fake_loss = fake_pred.mean()
	#
	# # Compute loss for gradient penalty.
	# beta = torch.rand(train_data.batch_size, 1, 1, 1).to(device)
	# x_hat = (beta * real_image.data + (1 - beta) * fake_image.data).requires_grad_(True)
	# d_x_hat_out = discriminator(x_hat, train_data.rindex, alpha_compo=alpha_compo)
	# d_loss_gp = gradient_penalty(d_x_hat_out, x_hat)
	#
	# # Backward and optimize.
	# d_loss = d_real_loss + d_fake_loss + 10.0 * d_loss_gp
	# d_loss.backward()
	# d_optimizer.step()
