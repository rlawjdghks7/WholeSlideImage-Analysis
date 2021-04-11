import numpy as np
import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from data_loader.dataset import Asan_train_dataset, Asan_test_dataset, Asan_offline, Asan_online
import matplotlib.pyplot as plt
import sys


def get_train_val_loader(root_dir, batch_size=4, augment=False, random_seed=1, 
			shuffle=True, validation_ratio=0.3, pin_memory=False,
			num_workers=4, slide_level=0, p_size=256):
	if augment:
		data_transforms = transforms.Compose([
				# transforms.Resize(256),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])
	else:
		data_transforms = transforms.Compose([
				# transforms.Resize(256),
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])

	train_datasets = Asan_train_dataset(root_dir, train_mode='train', 
					slide_level=slide_level, p_size=p_size,
					transforms=data_transforms)

	val_datasets = Asan_train_dataset(root_dir, train_mode='val', 
					slide_level=slide_level, p_size=p_size,
					transforms=data_transforms)

	dataloaders = {}
	dataloaders['train'] = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size,
									shuffle=True, num_workers=num_workers, 
									pin_memory=pin_memory)
	dataloaders['val'] = torch.utils.data.DataLoader(val_datasets, batch_size=1,
									shuffle=True, num_workers=num_workers,
									pin_memory=pin_memory)

	dataset_sizes = {}
	dataset_sizes['train'] = len(train_datasets)
	dataset_sizes['val'] = len(val_datasets)

	return dataloaders, dataset_sizes


def get_test_loader(root_dir, batch_size=1, pin_memory=False,
					num_workers=4, slide_level=0, p_size=256):

	data_transforms = transforms.Compose([
		# transforms.Resize(256),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	test_datasets = Asan_offline(root_dir, train_mode='test', 
					slide_level=slide_level, p_size=p_size,
					transforms=data_transforms)
	test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size,
									num_workers=num_workers, pin_memory=pin_memory)
	dataset_sizes = len(test_datasets)

	return test_loader, dataset_sizes


def get_test_map_loader(root_dir, slide_name, batch_size=1, pin_memory=False,
						num_workers=4, slide_level=0, p_size=256):
	
	data_transforms = transforms.Compose([
		# transforms.Resize(256),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	test_datasets = Asan_test_dataset(root_dir, slide_name, train_mode='test',
						slide_level=slide_level, p_size=p_size,
						transforms=data_transforms)
	test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size,
									num_workers=num_workers, pin_memory=pin_memory)

	dataset_sizes = len(test_datasets)
	return test_loader, dataset_sizes


def get_generate_map_loader(slide_path, batch_size=1, pin_memory=False,
							num_workers=4, slide_level=0, p_size=256):
	data_transforms = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	slide_datasets = Asan_online(slide_path, slide_level=slide_level,
								p_size=p_size, transforms=data_transforms)
	slide_loader = torch.utils.data.DataLoader(slide_datasets, batch_size=batch_size,
									num_workers=num_workers, pin_memory=pin_memory)
	
	return slide_loader	


def imshow(inp, title=None):
	inp = inp.numpy().transpose((1, 2, 0))
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	inp = std * inp + mean
	inp = np.clip(inp, 0, 1)
	plt.imshow(inp)
	if title is not None:
		plt.title(title)
	plt.pause(0.001)


def main():
	# dataloaders, dataset_sizes = get_train_val_loader('data/Kidney/Tumor_Normal')

	# inputs, classes = next(iter(dataloaders['train']))
	# out = torchvision.utils.make_grid(inputs)

	# print(dataset_sizes)
	slide_loader = get_generate_map_loader('data/Kidney/MSI_MSS/test/MSI/[PATHOLOGY]10S-049265-B1-T.tif')
	print(len(slide_loader))
	

if __name__ == '__main__':
	main()