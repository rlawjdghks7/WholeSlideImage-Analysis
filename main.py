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
from optparse import OptionParser
from data_loader import get_train_val_loader, get_test_loader, get_test_map_loader
from network import initialize_model
import openslide



def train(model, dataloaders, dataset_sizes, device, 
		criterion, optimizer, scheduler, num_epochs=100):
	since = time.time()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs-1))
		print('-'*10)

		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			if phase == 'train':
				scheduler.step()
				model.train()
			else:
				model.eval()

			running_loss = 0.0
			running_corrects = 0.0

			for inputs, labels, in dataloaders[phase]:
				inputs = inputs.to(device)
				labels = labels.to(device)
				print(len(inputs))
				
				optimizer.zero_grad()

				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs)
					_, preds = torch.max(outputs, 1)
					loss = criterion(outputs, labels)

					if phase == 'train':
						loss.backward()
						optimizer.step()

				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects.double() / dataset_sizes[phase]

			print('{} Loss : {:.4f} Acc: {:.4f}'.format(
				phase, epoch_loss, epoch_acc))

			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	# model.load_state_dict(best_model_wts)
	return model

def test(model, dataloaders, dataset_sizes, device,
		criterion):
	was_training = model.training
	model.eval()

	running_loss = 0.0
	running_corrects = 0

	with torch.no_grad():
		for inputs, labels, mask_x, mask_y in dataloaders:
			inputs = inputs.to(device)
			labels = labels.to(device)
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)
			loss = criterion(outputs, labels)

			running_loss += loss.item() * inputs.size(0)
			running_corrects += torch.sum(preds == labels.data)
			# print('preds is :{}, label is :{}'.format(preds, labels.data))
			# print('loss is :{}, corrects is :{}'.format(loss, preds==labels.data))
		total_loss = running_loss /dataset_sizes
		total_acc = running_corrects.double() / dataset_sizes
		print('{} Loss : {:.4f} ACC: {:.4f}'.format('test', total_loss, total_acc))



def save_model(model, model_dir, model_name):
	path = os.path.join(model_dir, model_name)
	if not os.path.isdir(model_dir):
		os.mkdir(model_dir)

	torch.save(model, path)


def get_args():
	parser = OptionParser()
	parser.add_option('-d', '--NAS_dir', dest='NAS_dir', default='/media/nas_mispl/dataset/PATHOLOGY_ANALYSIS/Asan/jh_experiments',
					  help='NAS workspace')
	parser.add_option('-t', '--data_type', dest='data_type', default='MSIH_MSS')
	parser.add_option('-n', '--network', dest='network', default='resnet',
					  help='network name')
	parser.add_option('-m', '--mode', dest='mode', default='train',
	                  help='training mode')
	parser.add_option('-o', '--online', dest='online', default=False,
					  help='patch data loading mode')
	parser.add_option('-l', '--slide_level', dest='slide_level', default=0, type='int',
					  help='set slide level')
	parser.add_option('-s', '--p_size', dest='p_size', default=256, type='int',
					  help='set patch size')

	parser.add_option('-e', '--epochs', dest='epochs', default=10, type='int',
	                  help='number of epochs')
	parser.add_option('-b', '--batch_size', dest='batch_size', default=4,
	                  type='int', help='batch size')
	parser.add_option('-r', '--learning_rate', dest='learning_rate', default=0.0001,
	                  type='float', help='learning rate')

	(options, args) = parser.parse_args()
	return options


def main():
	args = get_args()
	print('-'*10, 'Args', '-'*10)
	print(args)
	print('-'*10, '----', '-'*10)

	root_dir = os.path.join(args.NAS_dir, 'data')
	root_dir = os.path.join(root_dir, args.data_type)

	model_name = os.path.basename(root_dir)+'_'+args.network+'_epochs:'+str(args.epochs)+\
							'_slide_level-'+str(args.slide_level)+\
							'_p_size-'+str(args.p_size)+'.md'
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	# print(device)
	# sys.exit()
	criterion = nn.CrossEntropyLoss()

	if args.mode == 'train':
		print('train mode start')
		model_conv, _, _ = initialize_model(args.network, 2)
		dataloaders, dataset_sizes = get_train_val_loader(root_dir, batch_size=args.batch_size,
									slide_level=args.slide_level,
									p_size=args.p_size)
		model_conv = model_conv.to(device)
		
		optimizer = optim.SGD(model_conv.parameters(), lr=args.learning_rate, momentum=0.9)
		scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

		model = train(model_conv, dataloaders, dataset_sizes, device,
							criterion, optimizer, scheduler, num_epochs=args.epochs)
		
		print('save model : {}'.format(os.path.join('model', model_name)))
		save_model(model, 'model', model_name)
	elif args.mode == 'test':
		print('test mode start')
		print('model name is:', model_name)
		model_conv = torch.load(os.path.join('model', model_name))
		dataloaders, dataset_sizes = get_test_loader(root_dir,
									slide_level=args.slide_level,
									p_size=args.p_size)
		test(model_conv, dataloaders, dataset_sizes, device,
							criterion)



if __name__ == '__main__':
	main()