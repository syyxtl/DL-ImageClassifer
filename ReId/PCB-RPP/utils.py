import numpy
import torch
import matplotlib 

import os
import argparse

from torchvision import datasets, models, transforms

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--save_dir', default='./Models/', type=str, help='save model dir')
parser.add_argument('--data_dir', default='./Market/pytorch', type=str, help='training dir path')
parser.add_argument('--batch_size', default=2, type=int, help='batch_size')
parser.add_argument('--RPP', default=True, action='store_true', help='use RPP')
args = parser.parse_args()

SAVE_DIR = args.save_dir
DATA_DIR = args.data_dir
BATCHSIZE = args.batch_size
RPP = args.RPP

inputs = (384, 128)

def load_data():
	transform_train_list = [
		transforms.Resize( inputs , interpolation=3),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]

	transform_val_list = [
		transforms.Resize(size=inputs, interpolation=3), 
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]

	data_transforms = {
		'train': transforms.Compose(transform_train_list),
		'val': transforms.Compose(transform_val_list),
	}
	#dataset reader
	imagedatasets = {}
	imagedatasets['train'] = datasets.ImageFolder(
		os.path.join(DATA_DIR, 'train_all'), data_transforms['train']
	)
	imagedatasets['val'] = datasets.ImageFolder(
		os.path.join(DATA_DIR, 'val'), data_transforms['val']
	)
	# data reader
	dataloaders = {}
	dataloaders['train'] = torch.utils.data.DataLoader(
		imagedatasets['train'], batch_size=BATCHSIZE, shuffle=True, num_workers=0
	)  
	dataloaders['val'] = torch.utils.data.DataLoader(
		imagedatasets['val'], batch_size=BATCHSIZE, shuffle=True, num_workers=0
	)  
	# train size
	datasize = {}
	datasize['train'] = len(imagedatasets['train'])
	datasize['val'] = len(imagedatasets['val'])
	datasize['class'] = imagedatasets['train'].classes
	
	return {
		"train": dataloaders['train'], 
		"val": dataloaders['val'], 
		"train_size": datasize['train'], 
		"val_size": datasize['val'], 
		"class": datasize['class']
	}

def load_data_qg():
	data_transforms = transforms.Compose([
		transforms.Resize(inputs, interpolation=3),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	#dataset reader
	imagedatasets = {}
	imagedatasets['gallery'] = datasets.ImageFolder(
		os.path.join(DATA_DIR, 'gallery'), data_transforms
	)
	imagedatasets['query'] = datasets.ImageFolder(
		os.path.join(DATA_DIR, 'query'), data_transforms
	)
	# data reader
	dataloaders = {}
	dataloaders['gallery'] = torch.utils.data.DataLoader(
		imagedatasets['gallery'], batch_size=BATCHSIZE//2, shuffle=False, num_workers=0
	)  
	dataloaders['query'] = torch.utils.data.DataLoader(
		imagedatasets['query'], batch_size=BATCHSIZE//2, shuffle=False, num_workers=0
	)  
	# train size
	datasize = {}
	datasize['gallery'] = len(imagedatasets['gallery'])
	datasize['query'] = len(imagedatasets['query'])
	datasize['class'] = imagedatasets['query'].classes
	
	return {
		"gallery": dataloaders['gallery'], 
		"query": dataloaders['query'], 
		"class": datasize['class'], 
	}, imagedatasets

def load_data_ot():
	data_transforms = transforms.Compose([
		transforms.Resize(inputs, interpolation=3),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	#dataset reader
	imagedatasets = {}
	imagedatasets['one'] = datasets.ImageFolder(
		os.path.join(DATA_DIR, 'one'), data_transforms
	)
	imagedatasets['two'] = datasets.ImageFolder(
		os.path.join(DATA_DIR, 'two'), data_transforms
	)
	# data reader
	dataloaders = {}
	dataloaders['one'] = torch.utils.data.DataLoader(
		imagedatasets['one'], batch_size=1, shuffle=False, num_workers=0
	)  
	dataloaders['two'] = torch.utils.data.DataLoader(
		imagedatasets['two'], batch_size=1, shuffle=False, num_workers=0
	)  
	# train size
	datasize = {}
	datasize['one'] = len(imagedatasets['one'])
	datasize['two'] = len(imagedatasets['two'])
	
	return {
		"one": dataloaders['one'], 
		"two": dataloaders['two'], 
	}, imagedatasets