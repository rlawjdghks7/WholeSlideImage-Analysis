import os, cv2, sys
import openslide
import glob
import numpy as np
import shutil

from optparse import OptionParser
from xml.etree.ElementTree import parse
from PIL import Image
import openpyxl

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


def generate_patch_stride(slide_path, xml_path, patch_dir, slide_level=0, 
						p_size=256, resolution=0, n_patches=3000, 
						level=6):
	'''make patch img

	Args:
		slide_path	: full name of slide file path (dir/dir/slide)
		xml_path	: full name of xml file path (dir/dir/xml)
		patch_dir	: where save patch images
	'''
	slide_name = os.path.basename(slide_path).replace('.tif', '')
	save_patch_dir = os.path.join(patch_dir, slide_name)

	if not os.path.exists(save_patch_dir):
		print(slide_name, 'is generating...')
		os.makedirs(save_patch_dir)
	else:
		print(slide_name, 'is exist, remove and rebuild that...')
		shutil.rmtree(save_patch_dir)
		os.makedirs(save_patch_dir)

	slide = openslide.OpenSlide(slide_path)

	tissue_mask = get_tissue_mask(slide, level=level)
	tumor_mask = get_tumor_mask(slide, xml_path, level=level)
	step = int(p_size//(2**(level-slide_level)))
	width, height = np.array(slide.level_dimensions[slide_level])//p_size
	patch_cnt = 0
	
	for i in range(width):
		for j in range(height):
			tissue_mask_sum = tissue_mask[step*j:step*(j+1),
										step*i:step*(i+1)].sum()
			tumor_mask_sum = tumor_mask[step*j:step*(j+1),
										step*i:step*(i+1)].sum()

			mask_max = step*step*255
			tumor_area_ratio = tumor_mask_sum/mask_max
			tissue_area_ratio = tissue_mask_sum/mask_max
			if tumor_area_ratio > 0.7 and tissue_area_ratio > 0.7:
				x, y = p_size*i*(2**slide_level), p_size*j*(2**slide_level)
				patch_name = slide_name+'_'+str(x//(2**slide_level))+'_'+str(y//(2**slide_level))+'.png'
				patch_img = np.array(slide.read_region((x, y), slide_level, (p_size,p_size)))

				patch_img = cv2.cvtColor(patch_img, cv2.COLOR_RGBA2RGB)
				patch_img = cv2.cvtColor(patch_img, cv2.COLOR_RGB2BGR)
				cv2.imwrite(os.path.join(save_patch_dir, patch_name), patch_img)
				patch_cnt += 1
			if patch_cnt == n_patches:
				break
	print('number of patch is {}'.format(patch_cnt))



def MSIH_MSS_type(NAS_dir, MSIH_paths_dic, MSS_paths_dic,
				train_mode='train', patch_mode='stride', 
				slide_level=0, p_size=256, n_patches=3000): # extract patches for train data

	data_dir = os.path.join(NAS_dir, 'data')
	if not os.path.exists(data_dir):
		os.makedirs(data_dir)

	type_dir = os.path.join(data_dir, 'MSIH_MSS')
	if not os.path.exists(type_dir):
		os.makedirs(type_dir)

	patch_dir_name = train_mode + '_patch_' + patch_mode +\
						'_slide_level-' + str(slide_level) + \
						'_p_size-' + str(p_size)
	patch_dir = os.path.join(type_dir, patch_dir_name)
	if not os.path.exists(patch_dir):
		os.makedirs(patch_dir)

	MS_patch_dir = {x: os.path.join(patch_dir, x)
				for x in ['MSIH', 'MSS']}

	if not os.path.exists(MS_patch_dir['MSIH']):
		os.makedirs(MS_patch_dir['MSIH'])
	if not os.path.exists(MS_patch_dir['MSS']):
		os.makedirs(MS_patch_dir['MSS'])

	MS_dir = {x: os.path.join(data_dir, x)
			for x in ['MSIH', 'MSS']}

	print('#generate MSIH patches')
	for MSIH_path in MSIH_paths_dic[train_mode]:
		xml_path = MSIH_path.replace('.tif', '.xml')
		if patch_mode =='stride':
			generate_patch_stride(MSIH_path, xml_path, MS_patch_dir['MSIH'],
									slide_level=slide_level, p_size=p_size,
									n_patches=n_patches)

	print('#genearate MSS patches')
	for MSS_path in MSS_paths_dic[train_mode]:
		xml_path = MSS_path.replace('.tif', '.xml')
		if patch_mode =='stride':
			generate_patch_stride(MSS_path, xml_path, MS_patch_dir['MSS'],
									slide_level=slide_level, p_size=p_size,
									n_patches=n_patches)



def split_slide(args, train_idx=9, val_idx=1): # test is residue
	slide_paths = glob.glob(args.slide_dir+'/*.tif')
	xml_paths = glob.glob(args.slide_dir+'/*.xml')
	all_files = os.listdir(args.slide_dir)

	xlsx_path = os.path.join(args.NAS_dir, args.xlsx_name)
	wb = openpyxl.load_workbook(xlsx_path)
	ws = wb.active

	path_dic = {'MSIH':[], 'MSS':[]}
	for r in ws.rows:
		if r[0].value in all_files and (r[3].value=='MSIH' or r[3].value=='MSS') and int(r[4].value)==1:
			path_dic[r[3].value].append(os.path.join(args.slide_dir, r[0].value)) # r[3].value is MSIH or MSS

	MSIH_paths_dic = {}
	MSIH_paths_dic['train'] = path_dic['MSIH'][:train_idx]
	MSIH_paths_dic['val'] = path_dic['MSIH'][train_idx:train_idx+val_idx]
	MSIH_paths_dic['test'] = path_dic['MSIH'][train_idx+val_idx:]

	MSS_paths_dic = {}
	MSS_paths_dic['train'] = path_dic['MSS'][:train_idx]
	MSS_paths_dic['val'] = path_dic['MSS'][train_idx:train_idx+val_idx]
	MSS_paths_dic['test'] = path_dic['MSS'][train_idx+val_idx:]

	return MSIH_paths_dic, MSS_paths_dic



def read_xlsx(xlsx_path):
	wb = openpyxl.load_workbook(xlsx_path)
	ws = wb.active
	for r in ws.rows:
		print(r[0].row, r[0].value, r[1].value, r[2].value, r[3].value)
	sys.exit()



def get_args():
	parser = OptionParser()
	parser.add_option('-d', '--NAS_dir', dest='NAS_dir',
						default='/media/nas_mispl/PATHOLOGY_ANALYSIS/Asan/jh_experiments',
						help='NAS workspace')
	parser.add_option('-m', '--train_mode', dest='train_mode',
						default='train',
						help='train or val or test mode')
	parser.add_option('--slide_dir', dest='slide_dir',
						default='/media/jh/SAMSUNG/PATHOLOGY_ANALYSIS/Asan/Anonymized_Img/Anonymized_Colon_KMJ_SET2_20190523',
						help='External hard disk, there are slide images')
	parser.add_option('-x', '--xlsx_name', dest='xlsx_name',
						default='anonymized_file_list.xlsx')
	parser.add_option('-p', '--patch_mode', dest='patch_mode',
						default='stride',
						help='stride or random mode')
	parser.add_option('-l', '--slide_level', dest='slide_level',
						default=0,
						help='set slide level, default is 0')
	parser.add_option('-s', '--p_size', dest='p_size',
						default=256,
						help='set patch size, default is 256')
	parser.add_option('-n', '--n_patches', dest='n_patches',
						default=3000,
						help='set number of patches, default is 3000')
	(options, args) = parser.parse_args()

	return options


def main():
	''' for generate patch images
	'''
	args = get_args()
	print('-'*10, 'Args', '-'*10)
	print(args)
	print('-'*10, '----', '-'*10)

	# xlsx_path = os.path.join(args.NAS_dir, args.xlsx_name)
	# read_xlsx(xlsx_path)


	MSIH_paths_dic, MSS_paths_dic = split_slide(args)

	# print(len(MSIH_paths_dic['train']), len(MSIH_paths_dic['val']), len(MSIH_paths_dic['test']))
	# print(len(MSS_paths_dic['train']), len(MSS_paths_dic['val']), len(MSS_paths_dic['test']))
	# if os.path.exists(MSIH_paths_dic['train'][0]):
		# print(MSIH_paths_dic['train'][0], 'file is exsits')
	# sys.exit()

	MSIH_MSS_type(args.NAS_dir, MSIH_paths_dic, MSS_paths_dic,
				args.train_mode, args.patch_mode,
				int(args.slide_level), int(args.p_size), int(args.n_patches))

	


if __name__ == '__main__':
	main()