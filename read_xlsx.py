import openpyxl
import os, sys
from optparse import OptionParser
import glob

def get_args():
	parser = OptionParser()
	parser.add_option('-n', '--nas_Asan_dir', dest='nas_Asan_dir',
				default='/media/nas_mispl/PATHOLOGY_ANALYSIS/Asan',
				help='NAS Asan Directory')
	parser.add_option('-e', '--external_hdd', dest='external_hdd',
				default='/media/jh/SAMSUNG/PATHOLOGY_ANALYSIS/Asan/Anonymized_Img',
				help='External hardisk, there are slide images')
	parser.add_option('-d', '--data_name', dest='data_name',
						default='anonymized_file_list.xlsx',
						help='Data directory path')
	parser.add_option('-t', '--data_type', dest='data_type',
						default='Tumor_Normal',
						help='Tumor_Normal or MSI_MSS')
	(options, args) = parser.parse_args()

	return options


def classify_Tumor_Normal(work_sheet):
	Tumor_list = []
	Normal_list = []

	for i, r in enumerate(work_sheet.rows):
		# if i >= 50 and i <= 245:
		# 	# print(r[0].value)
		# 	# sys.exit()
		if r[3].value == 'T':
			Tumor_list.append(r[0].value)
			# print(r[0].value, 'Tumor adding')
		elif r[3].value == 'N':
			# print(r[0].value, 'Normal adding')
			Normal_list.append(r[0].value)
		else:
			print(r[0].value, 'Nothing')

	return Tumor_list, Normal_list


def classify_MSIH_MSS(work_sheet):
	MSIH_list = []
	MSS_list = []

	for i, r in enumerate(work_sheet.rows):
		if i >= 50 and i <= 245:
			# print(r[0].value)
			# sys.exit()
			if r[4].value == 'MSIH':
				MSIH_list.append(r[0].value)
				# print(r[0].value, 'MSIH adding')
			elif r[4].value == 'MSS':
				# print(r[0].value, 'MSS adding')
				MSS_list.append(r[0].value)
			else:
				print(r[0].value, 'Nothing')

	return MSIH_list, MSS_list


def main():
	# nas_Asan_dir = '/media/nas_mispl/PATHOLOGY_ANALYSIS/Asan'
	args = get_args()
	nas_Asan_dir = args.nas_Asan_dir
	xlsx_path = os.path.join(nas_Asan_dir, args.data_name)
	# file_list = os.listdir(nas_Asan_dir)
	# print(file_list)
	wb = openpyxl.load_workbook(xlsx_path)

	ws = wb.active
	# for r in ws.rows:
	# 	print(r[0].row, r[0].value, r[1].value, r[2].value)

	# Tumor_list, Normal_list = classify_Tumor_Normal(ws)
	# print(len(Tumor_list), len(Normal_list))

	# MSIH_list, MSS_list = classify_MSIH_MSS(ws)
	# print(len(MSIH_list), len(MSS_list))

	external_hdd = args.external_hdd
	set2_dir = os.path.join(external_hdd, 'Anonymized_Colon_KMJ_SET2_20190523')
	# file_list = os.listdir(external_hdd)
	# for file in file_list:
	# 	if 'SET2' in file:
	# 		print(file)
	all_files = os.listdir(set2_dir)
	imgs = glob.glob(os.path.join(set2_dir, '*.tif'))
	xmls = glob.glob(os.path.join(set2_dir, '*.xml'))
	# xmls = os.listdir(set2_dir)
	print(len(imgs))
	print(len(xmls))
	sys.exit()
	tumor_list = []
	normal_list = []
	for img in imgs:
		if 'T' in os.path.basename(img):
			# print("tumor img", img)
			tumor_list.append(img)
		elif 'N' in os.path.basename(img):
			# print('normal img', img)
			normal_list.append(img)
		# else:
			# print("error", img)
	print(len(tumor_list))
	print(len(normal_list))

	# cnt = 0
	# for tumor in tumor_list:
	# 	if not (os.path.basename(tumor).replace('.tif', '.xml') in xmls):
	# 		print(os.path.basename(tumor))
	# 		cnt += 1
	# 		print(cnt)

	for xml in xmls:
		if not os.path.exists(xml.replace('.xml', '.tif')):
			print(xml)


if __name__ == '__main__':
	main()