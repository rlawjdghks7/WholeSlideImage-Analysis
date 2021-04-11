import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import torch.nn.functional as F

import time
import os, sys, cv2
import copy
import glob
import openslide
from optparse import OptionParser
from data_loader import get_generate_map_loader
from network import initialize_model

import matplotlib.pyplot as plt

def visualize_img(img):
	# print(img.cpu().numpy().shape)
	img = img[0].cpu().numpy().transpose((1, 2, 0))
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	img = std * img + mean
	img = np.clip(img, 0, 1)
	plt.imshow(img)
	plt.show()

def generate_map(model, slide_path,  save_dir, dataloaders, device, 
				slide_level=0, p_size=256, level=6):
	was_training = model.training
	model.eval()

	slide_name = os.path.basename(slide_path)
	slide = openslide.OpenSlide(slide_path)
	width, height = slide.level_dimensions[level]
	probs_map = np.zeros((height, width))
	# print(slide.level_dimensions[level])
	# print(probs_map.shape)
	slide_lv = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]))
	slide_lv = cv2.cvtColor(slide_lv, cv2.COLOR_RGBA2RGB)
	slide_lv = cv2.cvtColor(slide_lv, cv2.COLOR_RGB2BGR)
	step = int(p_size//(2**(level-slide_level)))
	# print(step)
	# sys.exit()
	with torch.no_grad():
		for img, x, y, mask_x, mask_y in dataloaders:
			img = img.to(device)

			outputs = model(img)
			_, preds = torch.max(outputs, 1)
			probs_map[mask_x:mask_x+step, mask_y:mask_y+step] = \
						preds.item()
			# print(preds.item())
			# print('prediction result is :{}'.format(preds))
			# print(x, y)
			# visualize_img(img)

			# probs = F.softmax(outputs, dim=1).data.squeeze()[1]
			# probs_map[mask_x:mask_x+step, mask_y:mask_y+step] = \
			# 			int(probs.item()>0.5)
	heat_map = np.array(probs_map*255, dtype=np.uint8)
	heat_map = cv2.cvtColor(heat_map, cv2.COLOR_GRAY2RGB)
	heat_map[:,:,0] = 0
	heat_map[:,:,1] = 0
	result = np.array(slide_lv*0.7 + heat_map*0.3, dtype=np.uint8)
	cv2.imwrite(os.path.join(save_dir, slide_name), result)

def get_slide_paths(root_dir, mode='test'):
	data_type = os.path.basename(root_dir).split('_')
	mode_dir = os.path.join(root_dir, mode)
	type_dir = {x: os.path.join(mode_dir, x)
					for x in data_type}
	# print(type_dir)
	# sys.exit()

	slide_paths = []
	for x in data_type:
		slide_path = glob.glob(type_dir[x] + '/*tif')
		slide_paths += slide_path

	return slide_paths


def get_args():
	parser = OptionParser()
	parser.add_option('-d', '--data_dir', dest='data_dir', default='data/Kidney/MSI_MSS',
					  help='Data directory path')
	parser.add_option('-n', '--network', dest='network', default='resnet',
					  help='network name')
	parser.add_option('-m', '--mode', dest='mode', default='test',
	                  help='training mode')
	parser.add_option('-o', '--online', dest='online', default=False,
					  help='patch data loading mode')
	parser.add_option('-l', '--slide_level', dest='slide_level', default=0,
					  help='set slide level')
	parser.add_option('-s', '--p_size', dest='p_size', default=256,
					  help='set patch size')

	parser.add_option('-e', '--epochs', dest='epochs', default=10, type='int',
	                  help='number of epochs')
	parser.add_option('-b', '--batch-size', dest='batchsize', default=4,
	                  type='int', help='batch size')


	(options, args) = parser.parse_args()
	return options


def main():
	args = get_args()
	print('-'*10, 'Args', '-'*10)
	print(args)
	print('-'*10, '----', '-'*10)

	title_name = os.path.basename(args.data_dir)+'_'+args.network+'_epochs:'+str(args.epochs)+\
							'_slide_level:'+str(args.slide_level)+\
							'_p_size:'+str(args.p_size)
	model_name = title_name+'.md'
	print('model name is :', model_name)
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	print('Generate Map!')
	if not os.path.exists('heat_map'):
		os.makedirs('heat_map')

	data_type = os.path.basename(args.data_dir)
	type_dir = os.path.join('heat_map', data_type)
	if not os.path.exists(type_dir):
		os.makedirs(type_dir)

	save_dir = os.path.join(type_dir, title_name)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	model_conv = torch.load(os.path.join('model', model_name))
	slide_paths = get_slide_paths(args.data_dir, mode=args.mode)
	# for p in slide_paths:
	# 	print(p)
	# sys.exit()
	for slide_path in slide_paths:
		slide_name = os.path.basename(slide_path)
		print('Generating', slide_name, '...')
		dataloaders = get_generate_map_loader(slide_path, 
						slide_level=int(args.slide_level), p_size=int(args.p_size))
		generate_map(model_conv, slide_path, save_dir, dataloaders, 
			device, slide_level=int(args.slide_level), p_size=int(args.p_size))
		print(slide_name,' was generated!')
		# sys.exit()


if __name__ == '__main__':
	main()