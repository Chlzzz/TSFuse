import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import time
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.utils.data import DataLoader
import loss
from torch.optim import Adam
from torch.autograd import Variable
import utils
from TSFuse import TSFuse
from args_fusion import args
import pytorch_msssim

from dataset import H5Dataset


EPSILON = 1e-5


batch_size = args.batch_size


trainloader = DataLoader(
	H5Dataset(r"/home/heu/MSRS_train_imgsize_128_stride_200.h5"),  # location of dadaset.h5
	batch_size=batch_size,
	shuffle=True,
	num_workers=0,
)

loader = {'train': trainloader, }


# load network model
model = TSFuse(config=[2,2,2]).cuda()
print(model)

if args.resume is not None:
	print('Resuming, initializing using weight from {}.'.format(args.resume))
	model.load_state_dict(torch.load(args.resume))
	
#print(model)
optimizer = Adam(model.parameters(), args.lr)
g_content_criterion = loss.g_content_loss().cuda()
grad = loss.grad().cuda()

if args.cuda:
	model.cuda()
	grad.cuda()
	

tbar = trange(args.epochs)
print('Start training.....')

count_loss = 0
Loss_SSIM = []
Loss_Texture =[]
Loss_Indensity = []
Loss_all = []

for e in tbar:
	print('Epoch %d.....' % e)

	# if e == 15:
	# 		optimizer = Adam(model.parameters(), args.lr / 2)
	# 		args.log_interval = 2 
	
	model.train()
	count = 0
	batches = len(trainloader)
	for batch, (img_vi, img_ir) in enumerate(loader['train']):
	
		count += 1
		optimizer.zero_grad()
		img_ir = Variable(img_ir, requires_grad=False)
		img_vi = Variable(img_vi, requires_grad=False)

		if args.cuda:
			img_ir = img_ir.cuda()
			img_vi = img_vi.cuda()


		# decode
		outputs = model(img_vi, img_ir)

		x_ir = Variable(img_ir.data.clone(), requires_grad=False)
		x_vi = Variable(img_vi.data.clone(), requires_grad=False)


		######################### LOSS FUNCTION #########################
		all_Texture_loss =0.
		all_SSIM_loss = 0.
		all_intensity_loss = 0.
		all_total_loss = 0.

		total_loss,SSIM_loss,Texture_loss,Intensity_loss = g_content_criterion(x_ir,x_vi,outputs)

		all_SSIM_loss += SSIM_loss.item()
		all_Texture_loss += Texture_loss.item()
		all_intensity_loss += Intensity_loss.item()
		all_total_loss += total_loss.item()

		total_loss.backward()
		optimizer.step()

		if (batch + 1) % args.log_interval == 0:
			mesg = "{}\tEpoch {}:\t[{}/{}]\t SSIM LOSS: {:.6f}\t Texture LOSS: {:.6f}\t Intensity LOSS: {:.6f}\t total: {:.6f}".format(
				time.ctime(), e + 1, count, batches,
							all_SSIM_loss / args.log_interval,
							all_Texture_loss / args.log_interval,
							all_intensity_loss / args.log_interval,
							all_total_loss / args.log_interval
			)
			tbar.set_description(mesg)
			Loss_SSIM.append(all_SSIM_loss / args.log_interval)
			Loss_Texture.append(all_Texture_loss / args.log_interval)
			Loss_Indensity.append(all_intensity_loss / args.log_interval)
			Loss_all.append(all_total_loss / args.log_interval)
			count_loss = count_loss + 1

	if (e+1) % args.log_interval == 0:
		# save model
		model.eval()
		model.cuda()
		fuse_model_filename = "fuse_Epoch_" + str(e) + ".model"
		fuse_model_path = os.path.join(args.save_model_dir, fuse_model_filename)
		torch.save(model.state_dict(), fuse_model_path)

# SSIM loss
loss_data_SSIM = Loss_SSIM
loss_filename_path = 'final_SSIM.mat'
scio.savemat(args.save_loss_dir + loss_filename_path, {'final_loss_SSIM': loss_data_SSIM})

# Indensity loss
loss_data_Indensity = Loss_Indensity
loss_filename_path = "final_Indensity.mat"
scio.savemat(args.save_loss_dir + loss_filename_path, {'final_loss_Indensity': loss_data_Indensity})

# Indensity loss
loss_data_Texture = Loss_Texture
loss_filename_path = "final_Texture.mat"
scio.savemat(args.save_loss_dir + loss_filename_path, {'final_loss_Texture': loss_data_Texture})

loss_data = Loss_all
loss_filename_path = "final_all.mat"
scio.savemat(args.save_loss_dir + loss_filename_path, {'final_loss_all': loss_data})

# save model
model.eval()
model.cpu()
save_model_filename = "final_epoch.model"
torch.save(model.state_dict(), save_model_filename)

print("\nDone, trained model saved at", save_model_filename)


