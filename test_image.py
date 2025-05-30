# test phase
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision import datasets, transforms
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from args_fusion import args
import numpy as np
import time
import os
import utils

from TSFuse import TSFuse
from evaluator import evaluate

def load_model(path):
	nest_model = TSFuse()
	nest_model.load_state_dict(torch.load(path))

	#print(parameter_count_table(nest_model))
	para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))

	nest_model.eval()
	nest_model.cuda()

	return nest_model


def _generate_fusion_image(model, img_ir, img_vi):
	_, _, h_old, w_old = img_ir.size()
	xx = 16
	if h_old % xx != 0 or w_old % xx != 0:
		h_new = int(torch.ceil(torch.Tensor([h_old]) / (xx))) * (xx)
		w_new = int(torch.ceil(torch.Tensor([w_old]) / (xx))) * (xx)

		padding_h = h_new - h_old
		padding_w = w_new - w_old

		img_ir = torch.nn.functional.pad(img_ir, (0, padding_w, 0, padding_h))
		img_vi = torch.nn.functional.pad(img_vi, (0, padding_w, 0, padding_h))


	img_fusion = model(img_vi, img_ir)

	if h_old % xx != 0 or w_old % xx != 0:
		top = 0
		bottom = img_fusion.shape[2] - padding_h
		left = 0
		right = img_fusion.shape[3] - padding_w
		img_fusion = img_fusion[:, :, top:bottom, left:right]

	return img_fusion


def run_demo(model, infrared_path, visible_path, output_path_root,img_name, mode, mode2):
	ir_img = utils.get_test_images(infrared_path, height=None, width=None, mode=mode)
	vis_img = utils.get_test_images(visible_path, height=None, width=None, mode=mode2)
	if args.cuda:
		ir_img = ir_img.cuda()
		vis_img = vis_img.cuda()
	ir_img = Variable(ir_img, requires_grad=False)
	vis_img = Variable(vis_img, requires_grad=False)

	img_fusion = _generate_fusion_image(model, ir_img, vis_img) * 255

	file_name = str(img_name)
	output_path = output_path_root + file_name
	# # save images
	if args.cuda:
		img = img_fusion.cpu().clamp(0, 255).data[0].numpy()
	else:
		img = img_fusion.clamp(0, 255).data[0].numpy()

	img = img.transpose(1, 2, 0)
	if mode2 == 'YCbCr':
		xx = Image.open(visible_path).convert('YCbCr')
		y, cb, cr = xx.split()
		img = transforms.ToPILImage()(np.uint8(img))
		img = Image.merge('YCbCr', [img, cb, cr]).convert('RGB')
		img.save(output_path)
	else:
		img = transforms.ToPILImage()(np.uint8(img))
		img = img.convert('L')
		img.save(output_path)

def main():
	# run demo

	# in_c = 1 # generate gray image
	in_c = 3
	if in_c == 1:
		# test_path = r"/media/nh/DataSets/MSRS-main/test/"
		# output_path = r"/media/nh/DataSets/MSRS-main/test/output/"
		test_path = r"./images/MSRS/"
		output_path = './images/MSRS/fusion_out/'
	else :
		# test_path = r"./images/TNO/"
		# output_path = r'./images/TNO/fusion_out/'
		test_path = r"./images/MSRS/"
		output_path = './images/MSRS/fusion_out/'
		# test_path = r"/media/heu/DataSets/MSRS-main/test/"
		# output_path = r"/media/heu/DataSets/MSRS-main/test/output/"

	if os.path.exists(output_path) is False:
		os.mkdir(output_path)

	if in_c == 1:
		out_c = in_c
		mode = 'L'
		mode2 = 'L'
		model_path = args.model_path_gray_test
	else:
		out_c = 1
		mode = 'L'
		mode2 = 'YCbCr'
		#mode2 = 'L'
		model_path = args.model_path_gray_test
		#print(model_path)
	start = time.time()
	with torch.no_grad():
		model = load_model(model_path)
		for img_name in os.listdir(os.path.join(test_path,"ir")):
			print(img_name)
			# infrared_path = test_path +'IR/' + str(index) + '.png'
			# visible_path = test_path + 'VI_RGB/' + str(index) + '.png'
			infrared_path = test_path +'ir/' + img_name
			visible_path = test_path + 'vi/' + img_name

			run_demo(model, infrared_path, visible_path, output_path, img_name, mode, mode2)
	print('Done......')
	end = time.time()
	print(end - start, 's')

	# root_path = '/media/heu/DataSets/MSRS-main/test'
	# evaluate(root_path=root_path)

if __name__ == '__main__':
	main()
