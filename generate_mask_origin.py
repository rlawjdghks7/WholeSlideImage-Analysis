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
from xml.etree.ElementTree import parse
from PIL import Image

def read_xml(xml_dir, level=6):
	''' read xml files witch has tumor corrdinates list
		return coordinates of tumor areas

	Args:
		xml_dir	: full name of xml file(.xml)
		level	: level of mask
	'''

	xml = parse(xml_dir).getroot()

	coors_list = []
	coors = []
	for areas in xml.iter('Coordinates'):
		for area in areas:
			coors.append([round(float(area.get('X'))/(2**level)),
							round(float(area.get('Y'))/(2**level))])

		coors_list.append(coors)
		coors = []
	return np.array(coors_list)


def get_tissue_mask(slide, level=6):
	''' generate tissue mask witch has binary tissue area
		return binary image

	Args:
		slide 	: whole slide image
		level 	: level of mask
	'''

	slide_lv = slide.read_region((0, 0), level,
								slide.level_dimensions[level])
	slide_lv = cv2.cvtColor(np.array(slide_lv), cv2.COLOR_RGBA2RGB)
	hsv = cv2.cvtColor(slide_lv, cv2.COLOR_BGR2HSV)
	slide_lv = hsv[:, :, 1]
	_, tissue_mask = cv2.threshold(slide_lv, 0, 255,
								cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	return tissue_mask


def get_tumor_mask(slide, xml_path, level=6):
	''' generate tumor mask itch has binary tumor area
		return binary image

	Args:
		slide 		: whole slide image
		xml_path 	: full name of xml file(.xml)
		level 		: level of mask
	'''

	coors_list = read_xml(xml_path, level)
	tumor_mask = np.zeros(slide.level_dimensions[level][::-1])

	for coors in coors_list:
		cv2.drawContours(tumor_mask, np.array([coors]), -1, 255, -1)

	return tumor_mask


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


def generate_mask_origin(slide_path, save_path, slide_level=6):
	slide_name = os.path.basename(slide_path)
	slide = openslide.OpenSlide(slide_path)
	slide_lv = slide.read_region((0, 0), slide_level,
							slide.level_dimensions[slide_level])
	xml_path = slide_path.replace('.tif', '.xml')
	tumor_mask = np.array(get_tumor_mask(slide, xml_path), dtype=np.uint8)
	# print(tumor_mask.shape)
	# sys.exit()
	tumor_mask = cv2.cvtColor(tumor_mask, cv2.COLOR_GRAY2RGB)
	tumor_mask[:,:,0] = 0
	tumor_mask[:,:,1] = 0
	np_slide = np.array(slide_lv)
	np_slide = cv2.cvtColor(np_slide, cv2.COLOR_RGBA2RGB)
	np_slide = cv2.cvtColor(np_slide, cv2.COLOR_RGB2BGR)
	overlab = np.array(np_slide*0.7 + tumor_mask*0.3, dtype=np.uint8)
	origin_path = os.path.join(save_path, slide_name)
	mask_path = os.path.join(save_path, slide_name.replace('.tif', '_mask.tif'))

	slide_lv.save(origin_path)
	cv2.imwrite(mask_path, overlab)



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

	print('Generate mask_origin!')
	if not os.path.exists('mask_origin'):
		os.makedirs('mask_origin')

	data_type = os.path.basename(args.data_dir)
	type_dir = os.path.join('mask_origin', data_type)
	if not os.path.exists(type_dir):
		os.makedirs(type_dir)

	slide_paths = get_slide_paths(args.data_dir, mode=args.mode)
	for slide_path in slide_paths:
		slide_name = os.path.basename(slide_path)
		print('Generating', slide_name, '...')
		
		generate_mask_origin(slide_path, type_dir)
		print(slide_name,' was generated!')
		# sys.exit()


if __name__ == '__main__':
	main()