import os, cv2, sys
import openslide
import glob
import numpy as np
import shutil

from optparse import OptionParser
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


def Normal_patch_stride(slide_path, xml_path, Normal_patch_dir, 
						slide_level=0, p_size=256, resolution=0, patch_max_num=10000, 
						level=6):
	'''make patch img

	Args:
		slide_path	: full name of slide file path (dir/dir/slide)
		xml_path	: full name of xml file path (dir/dir/xml)
		patch_dir	: where save patch images
	'''
	slide_name = os.path.basename(slide_path).replace('.tif', '')
	save_patch_dir = os.path.join(Normal_patch_dir, slide_name)

	if not os.path.exists(save_patch_dir):
		print(slide_name, 'is generating...')
		os.makedirs(save_patch_dir)
	else:
		print(slide_name, 'is exist, remove and rebuild that...')
		shutil.rmtree(save_patch_dir)
		os.makedirs(save_patch_dir)

	slide = openslide.OpenSlide(slide_path)

	tissue_mask = get_tissue_mask(slide, level=level)
	step = int(p_size//(2**(level-slide_level)))
	width, height = np.array(slide.level_dimensions[slide_level])//p_size
	patch_cnt = 0

	if xml_path is None: # Normal slide
		for i in range(width):
			for j in range(height):
				tissue_mask_sum = tissue_mask[step*j:step*(j+1),
											step*i:step*(i+1)].sum()
				mask_max = step*step*255
				tissue_area_ratio = tissue_mask_sum/mask_max

				if tissue_area_ratio > 0.7:
					# set starting point
					x, y = p_size*i*(2**slide_level), p_size*j*(2**slide_level)

					patch_name = slide_name+'_'+str(x)+'_'+str(y)+'.png'
					patch_img = np.array(slide.read_region((x, y), slide_level, (p_size,p_size)))

					patch_img = cv2.cvtColor(patch_img, cv2.COLOR_RGBA2RGB)
					patch_img = cv2.cvtColor(patch_img, cv2.COLOR_RGB2BGR)
					cv2.imwrite(os.path.join(save_patch_dir, patch_name), patch_img)
					patch_cnt += 1

				if patch_cnt == patch_max_num:
					break
	print('number of patch is {}'.format(patch_cnt))


def Tumor_patch_stride(slide_path, xml_path, Tumor_patch_dir, Normal_patch_dir,
						slide_level=0, p_size=256, resolution=0, patch_max_num=10000, 
						level=6):
	'''make patch img

	Args:
		slide_path	: full name of slide file path (dir/dir/slide)
		xml_path	: full name of xml file path (dir/dir/xml)
		patch_dir	: where save patch images
	'''
	slide_name = os.path.basename(slide_path).replace('.tif', '')
	tumor_save_patch_dir = os.path.join(Tumor_patch_dir, slide_name)
	normal_save_patch_dir = os.path.join(Normal_patch_dir, slide_name)

	if not os.path.exists(tumor_save_patch_dir):
		print(slide_name, 'is generating...')
		os.makedirs(tumor_save_patch_dir)
	else:
		print(slide_name, 'is exist, remove and rebuild that...')
		shutil.rmtree(tumor_save_patch_dir)
		os.makedirs(tumor_save_patch_dir)
		
	if not os.path.exists(normal_save_patch_dir):
		os.makedirs(normal_save_patch_dir)
	else:
		shutil.rmtree(normal_save_patch_dir)
		os.makedirs(normal_save_patch_dir)


	slide = openslide.OpenSlide(slide_path)

	tissue_mask = get_tissue_mask(slide, level=level)
	step = int(p_size//(2**(level-slide_level)))
	width, height = np.array(slide.level_dimensions[slide_level])//p_size
	patch_cnt = 0

	if xml_path is not None: # Tumor slide
		tumor_mask = get_tumor_mask(slide, xml_path, level=level)
		
		for i in range(width):
			for j in range(height):
				tissue_mask_sum = tissue_mask[step*j:step*(j+1),
											step*i:step*(i+1)].sum()
				tumor_mask_sum = tumor_mask[step*j:step*(j+1),
											step*i:step*(i+1)].sum()

				mask_max = step*step*255
				tumor_area_ratio = tumor_mask_sum/mask_max
				tissue_area_ratio = tissue_mask_sum/mask_max

				if tissue_area_ratio > 0.7:
					x, y = p_size*i*(2**slide_level), p_size*j*(2**slide_level)
					patch_name = slide_name+'_'+str(x//(2**slide_level))+'_'+str(y//(2**slide_level))+'.png'
					patch_img = np.array(slide.read_region((x, y), slide_level, (p_size,p_size)))
					patch_img = cv2.cvtColor(patch_img, cv2.COLOR_RGBA2RGB)
					patch_img = cv2.cvtColor(patch_img, cv2.COLOR_RGB2BGR)

					if tumor_area_ratio > 0.5:
						save_patch_dir = tumor_save_patch_dir
					else:
						save_patch_dir = normal_save_patch_dir
					cv2.imwrite(os.path.join(save_patch_dir, patch_name), patch_img)
					patch_cnt += 1

				if patch_cnt == patch_max_num:
					break
	print('number of patch is {}'.format(patch_cnt))


def Tumor_Normal_type(NAS_dir, tumor_paths_dic, normal_paths_dic,
					train_mode='train', patch_mode='stride', slide_level=0, p_size=256,): # extract patches for train data
	patch_root_dir = os.path.join(NAS_dir, 'patch_data')
	if not os.path.exists(patch_root_dir):
		os.makedirs(patch_root_dir)

	patch_mode_dir = os.path.join(patch_root_dir, train_mode)
	if not os.path.exists(patch_mode_dir):
		os.makedirs(patch_mode_dir)

	

	patch_dir_name = train_mode + '_patch_' + patch_mode +\
						'_slide_level:' + str(slide_level) + \
						'_p_size:' + str(p_size)
	patch_dir = os.path.join(NAS_dir, patch_dir_name)

	# make patch dir
	if not os.path.exists(patch_dir):
		os.makedirs(patch_dir)

	Tumor_patch_dir = os.path.join(patch_dir, 'Tumor')
	Normal_patch_dir = os.path.join(patch_dir, 'Normal')

	if not os.path.exists(Tumor_patch_dir):
		os.makedirs(Tumor_patch_dir)
	if not os.path.exists(Normal_patch_dir):
		os.makedirs(Normal_patch_dir)

	TN_dir = {x: os.path.join(data_dir, x)
			for x in ['Tumor', 'Normal']}

	for x in ['Tumor', 'Normal']:
		slide_paths = glob.glob(TN_dir[x] + '/*.tif')
		xml_paths = []
		if x is 'Normal':
			for slide_path in slide_paths:
				if patch_mode == 'stride':
					Normal_patch_stride(slide_path, None, Normal_patch_dir,
									slide_level=slide_level, p_size=p_size)
		elif x is 'Tumor':
			for slide_path in slide_paths:
				xml_path = slide_path.replace('.tif', '.xml')
				if patch_mode == 'stride':
					Tumor_patch_stride(slide_path, xml_path, Tumor_patch_dir, Normal_patch_dir,
									slide_level=slide_level, p_size=p_size)


def split_slide(slide_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
	slide_paths = glob.glob(slide_dir+'/*.tif')
	xml_paths = glob.glob(slide_dir+'/*.xml')

	# print(len(slide_paths))
	# print(len(xml_paths))

	tumor_paths, normal_paths = [], []
	for slide_path in slide_paths:
		if 'N' in os.path.basename(slide_path):
			normal_paths.append(slide_path)

	for xml_path in xml_paths:
		tumor_path = xml_path.replace('.xml', '.tif')
		if os.path.exists(tumor_path):
			tumor_paths.append(tumor_path)

	tumor_len, normal_len = len(tumor_paths), len(normal_paths)
	train_idx = int(len(tumor_paths)*train_ratio)
	val_idx = train_idx + int(len(tumor_paths)*val_ratio)
	test_idx = val_idx + int(len(tumor_paths)*test_ratio)
	# print(train_idx, val_idx, test_idx)

	tumor_paths_dic = {}
	tumor_paths_dic['train'] = tumor_paths[:train_idx]
	tumor_paths_dic['val'] = tumor_paths[train_idx:val_idx]
	tumor_paths_dic['test'] = tumor_paths[val_idx:]

	train_idx = int(normal_len*train_ratio)
	val_idx = train_idx + int(normal_len*val_ratio)
	test_idx = val_idx + int(normal_len*test_ratio)

	normal_paths_dic = {}
	normal_paths_dic['train'] = normal_paths[:train_idx]
	normal_paths_dic['val'] = normal_paths[train_idx:val_idx]
	normal_paths_dic['test'] = normal_paths[val_idx:]
	
	# print('tumor_dic')
	# for x in ['train', 'val', 'test']:
	# 	print(len(tumor_paths_dic[x]))

	# print('normal_dic')
	# for x in ['train', 'val', 'test']:
	# 	print(len(normal_paths_dic[x]))

	return tumor_paths_dic, normal_paths_dic

def get_args():
	parser = OptionParser()
	parser.add_option('-d', '--NAS_dir', dest='NAS_dir',
						default='/media/nas_mispl/PATHOLOGY_ANALYSIS/Asan/jh_experiments',
						help='for save patch')
	parser.add_option('--slide_dir', dest='slide_dir',
						default='/media/jh/SAMSUNG/PATHOLOGY_ANALYSIS/Asan/Anonymized_Img/Anonymized_Colon_KMJ_SET2_20190523',
						help='External hard disk, there are slide images')
	parser.add_option('-m', '--train_mode', dest='train_mode',
						default='train',
						help='train or test mode')
	parser.add_option('-p', '--patch_mode', dest='patch_mode',
						default='stride',
						help='stride or random mode')
	parser.add_option('-l', '--slide_level', dest='slide_level',
						default=0,
						help='set slide level, default is 0')
	parser.add_option('-s', '--p_size', dest='p_size',
						default=256,
						help='set patch size, default is 256')
	(options, args) = parser.parse_args()

	return options


def main():
	''' for generate patch images
	'''
	args = get_args()
	print('-'*10, 'Args', '-'*10)
	print(args)
	print('-'*10, '----', '-'*10)

	tumor_paths_dic, normal_paths_dic = split_slide(args.slide_dir)

	Tumor_Normal_type(args.NAS_dir, tumor_paths_dic, normal_paths_dic,
						args.train_mode, args.patch_mode,
						int(args.slide_level), int(args.p_size))

	


if __name__ == '__main__':
	main()