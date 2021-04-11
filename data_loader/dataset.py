import torch.utils.data as data_utils

import numpy as np
import os, sys, cv2
import glob
import pathlib
from PIL import Image
from xml.etree.ElementTree import parse
import openslide

class Asan_train_dataset(data_utils.Dataset):
	def __init__(self, root_dir, train_mode='train', patch_mode='stride', 
				slide_level=0, p_size=256, transforms=None):
		# root_dir = /media/nas_mispl/PATHOLOGY_ANALYSIS/Asan/jh_experiments/data/MSIH_MSS
		self.train_mode = train_mode
		self.transforms = transforms
		self.data_type = os.path.basename(root_dir).split('_')
		patch_mode_dir = os.path.join(root_dir, train_mode+'_patch_'+patch_mode +\
						'_slide_level-' + str(slide_level) + \
						'_p_size-' + str(p_size))
		slide_patch_dirs = glob.glob(patch_mode_dir+'/'+self.data_type[0]+'/*') +\
							glob.glob(patch_mode_dir+'/'+self.data_type[1]+'/*')

		self.all_file_list = []
		for slide_patch_dir in slide_patch_dirs:
			path = pathlib.Path(slide_patch_dir)
			files = path.glob('*.png')

			file_list = list(files)
			self.all_file_list += file_list

	def _check_label(self, path):
		label = path.split('/')[-3]
		# print(label, self.data_type[0], self.data_type[1])
		if label == self.data_type[0]: # MSIH or Tumor
			return 1
		elif label == self.data_type[1]: # MSS or Normal
			return 0
		else:
			print('label was worng')
			sys.exit()

	def __getitem__(self, idx):
		path = str(self.all_file_list[idx])
		img = Image.open(path).convert('RGB')
		label = self._check_label(str(path))

		if self.transforms:
			img = self.transforms(img)
		
		return img, label

	def __len__(self):
		return len(self.all_file_list)

	def __str__(self):
		return "Asan Offline train_dataset"


class Asan_test_dataset(data_utils.Dataset): # for generating map
	def __init__(self, root_dir, slide_name, train_mode='test', patch_mode='stride', 
				slide_level=0, p_size=256, transforms=None):
		# root_dir = data/Kidney/Tumor_Normal
		self.slide_level = slide_level
		self.slide_name = slide_name
		self.train_mode = train_mode
		self.transforms = transforms
		self.data_type = os.path.basename(root_dir).split('_')
		self.p_size = p_size
		self.step = int(self.p_size//(2**(6-self.slide_level)))

		patch_mode_dir = os.path.join(root_dir, train_mode+'_patch_'+patch_mode +\
						'_slide_level-' + str(slide_level) + \
						'_p_size-' + str(p_size))
		slide_patch_dirs = [patch_mode_dir+'/'+self.data_type[0]+'/'+slide_name,
							patch_mode_dir+'/'+self.data_type[1]+'/'+slide_name]
		# x, y = self.p_size*i*(2**self.slide_level), self.p_size*j*(2**self.slide_level)

		self.all_file_list = []
		for slide_patch_dir in slide_patch_dirs:
			path = pathlib.Path(slide_patch_dir)
			
			files = path.glob('*.png')
			file_list = list(files)
			self.all_file_list += file_list

	def _check_label(self, path):
		label = path.split('/')[-3]
		if label == self.data_type[0]: # MSI or Tumor
			return 1
		elif label == self.data_type[1]: # MSS or Normal
			return 0
		else:
			print('label was worng')
			sys.exit()

	def __getitem__(self, idx):
		path = str(self.all_file_list[idx])

		slide_name = path.split('/')[-1].split('.')[0]
		x = int(slide_name.split('_')[-2])
		y = int(slide_name.split('_')[-1])
		mask_x = self.step*y/self.p_size/(2**self.slide_level)
		mask_y = self.step*x/self.p_size/(2**self.slide_level)

		img = Image.open(path).convert('RGB')
		label = self._check_label(str(path))
		# print(slide_name)
		# print(x, y)
		# sys.exit()

		if self.transforms:
			img = self.transforms(img)
		
		return img, label, mask_x, mask_y

	def __len__(self):
		return len(self.all_file_list)

	def __str__(self):
		return "Asan Offline Dataset Loader"


class Asan_offline(data_utils.Dataset):
	def __init__(self, root_dir, train_mode='train', patch_mode='stride', 
				slide_level=0, p_size=256, level=6, transforms=None):
		# root_dir = data/Kidney/Tumor_Normal
		self.slide_level = slide_level
		self.train_mode = train_mode
		self.transforms = transforms
		self.data_type = os.path.basename(root_dir).split('_')
		self.p_size = p_size
		self.step = int(self.p_size//(2**(level-self.slide_level)))

		patch_mode_dir = os.path.join(root_dir, train_mode+'_patch_'+patch_mode +\
						'_slide_level:' + str(slide_level) + \
						'_p_size:' + str(p_size))
		slide_patch_dirs = glob.glob(patch_mode_dir+'/'+self.data_type[0]+'/*') +\
							glob.glob(patch_mode_dir+'/'+self.data_type[1]+'/*')
		
		# x, y = self.p_size*i*(2**self.slide_level), self.p_size*j*(2**self.slide_level)

		self.all_file_list = []
		for slide_patch_dir in slide_patch_dirs:
			path = pathlib.Path(slide_patch_dir)

			files = path.glob('*.png')

			file_list = list(files)
			self.all_file_list += file_list

	def _check_label(self, path):
		label = path.split('/')[-3]
		# print(label, self.data_type[0], self.data_type[1])
		if label == self.data_type[0]: # MSI or Tumor
			return 1
		elif label == self.data_type[1]: # MSS or Normal
			return 0
		else:
			print('label was worng')
			sys.exit()

	def __getitem__(self, idx):
		path = str(self.all_file_list[idx])

		slide_name = path.split('/')[-1].split('.')[0]
		x = int(slide_name.split('_')[-2])
		y = int(slide_name.split('_')[-1])
		mask_x = self.step*y/self.p_size/(2**self.slide_level)
		mask_y = self.step*x/self.p_size/(2**self.slide_level)

		img = Image.open(path).convert('RGB')
		label = self._check_label(str(path))

		if self.transforms:
			img = self.transforms(img)
		
		return img, label, mask_x, mask_y

	def __len__(self):
		return len(self.all_file_list)

	def __str__(self):
		return "Asan Offline Dataset Loader"


class Asan_online(data_utils.Dataset): # for generate map
	def __init__(self, slide_path, 
				slide_level=0, p_size=256, transforms=None):
		self.slide = openslide.OpenSlide(slide_path)
		self.slide_path = slide_path
		self.xml_path = slide_path.replace('.tif', '.xml')
		self.slide_name = os.path.basename(slide_path)
		self.transforms = transforms
		self.class_name = slide_path.split('/')[-2]

		self.slide_level = slide_level
		self.p_size = p_size

		self.patch_list, self.x_y_position, self.mask_x_y_position = self._get_patch_stride()

	def __getitem__(self, idx):
		img = self.patch_list[idx]
		(x, y) = self.x_y_position[idx]
		(mask_x, mask_y) = self.mask_x_y_position[idx]

		if self.transforms:
			img = self.transforms(img)

		return img, x, y, mask_x, mask_y

	def __len__(self):
		return len(self.patch_list)

	def _read_xml(self, level=6):
		xml = parse(self.xml_path).getroot()

		coors_list = []
		coors = []
		for areas in xml.iter('Coordinates'):
			for area in areas:
				coors.append([round(float(area.get('X'))/(2**level)),
								round(float(area.get('Y'))/(2**level))])

			coors_list.append(coors)
			coors = []
		return np.array(coors_list)

	def _get_tumor_mask(self, level=6):
		coors_list = self._read_xml()
		tumor_mask = np.zeros(self.slide.level_dimensions[level][::-1])

		for coors in coors_list:
			cv2.drawContours(tumor_mask, np.array([coors]), -1, 255, -1)

		return tumor_mask

	def _get_patch_stride(self, level=6):
		slide = self.slide
		# tissue_mask = self._get_tissue_mask(level=level)
		tissue_mask = self._get_tumor_mask()
		# print(tissue_mask.shape)
		# sys.exit()
		step = int(self.p_size//(2**(level-self.slide_level)))
		width, height = np.array(slide.level_dimensions[self.slide_level])//self.p_size
		# print(width, height)
		# sys.exit()
		patch_cnt = 0
		patch_img_list = []
		x_y_position = []
		mask_x_y_position = []

		patch_cnt = 0
		for i in range(width):
			for j in range(height):
				tissue_mask_sum = tissue_mask[step*j:step*(j+1),
											step*i:step*(i+1)].sum()
				mask_max = step*step*255
				tissue_area_ratio = tissue_mask_sum/mask_max

				if tissue_area_ratio > 0.7:
					x, y = self.p_size*i*(2**self.slide_level), self.p_size*j*(2**self.slide_level)
					patch_img = np.array(slide.read_region((x, y), self.slide_level,
												(self.p_size, self.p_size)))
					patch_img = Image.fromarray(patch_img).convert('RGB')
					# patch_img = cv2.cvtColor(patch_img, cv2.COLOR_RGBA2RGB)
					# patch_img = cv2.cvtColor(patch_img, cv2.COLOR_RGB2BGR)
					
					# cv2.imshow('patch_img', patch_img)
					# print((x//(2**self.slide_level), y//(2**self.slide_level)))
					# cv2.waitKey()
					# sys.exit()
					
					patch_img_list.append(patch_img)
					x_y_position.append((y//(2**self.slide_level), x//(2**self.slide_level)))
					mask_x_y_position.append((step*j, step*i))
					patch_cnt += 1
					# print(patch_cnt)
				# if patch_cnt == 10:
				# 	break

		return patch_img_list, x_y_position, mask_x_y_position

	def _get_tissue_mask(self, level=6):
		slide_lv = self.slide.read_region((0, 0), level,
							self.slide.level_dimensions[level])
		slide_lv = cv2.cvtColor(np.array(slide_lv), cv2.COLOR_RGBA2RGB)
		hsv = cv2.cvtColor(slide_lv, cv2.COLOR_BGR2HSV)
		slide_lv = hsv[:, :, 1]
		_, tissue_mask = cv2.threshold(slide_lv, 0, 255,
									cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		return tissue_mask