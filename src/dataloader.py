
import random
import torch 
import os
import numpy as np

import constants as C 
import aux as A


def data_stream(data_seizure, data_nonseizure, balance, shuffle):

	if balance:
		if shuffle:
			random.shuffle(data_seizure)
			random.shuffle(data_nonseizure)
		data_all = list(zip(data_seizure, data_nonseizure))
		data_all = [x for y in data_all for x in y]
	
	else:
		data_all = data_nonseizure + data_seizure
		if shuffle:
			random.shuffle(data_all)

	for x in data_all:
		yield x


class SeizureDataset(torch.utils.data.IterableDataset):

	def __init__(self, data, balance = False, shuffle = True):
		(self.data_seizure, self.data_nonseizure) = data
		self.balance = balance
		self.shuffle = shuffle

	def __iter__(self):
		return data_stream(self.data_seizure, self.data_nonseizure, self.balance, self.shuffle)


def collate_fn(batch, normalize):

	data = [torch.tensor(d).float() for (d, _, _, _) in batch]
	labels = [torch.tensor(l).long() for (_, l, _, _) in batch]

	data = torch.stack(data)
	labels = torch.stack(labels).view(-1) 

	if normalize:
		mean = torch.mean(data, dim = 2, keepdim = True)
		std = torch.std(data, dim = 2, keepdim = True)
		std[std == 0] = 1

		data = (data - mean) / std

	return data, labels


def get_dataloader(data, balance, normalize, batch_size, shuffle):

	iterable_dataset = SeizureDataset(data, balance = balance, shuffle = shuffle) 
	loader = torch.utils.data.DataLoader(iterable_dataset, collate_fn = lambda x: collate_fn(x, normalize = normalize), batch_size = batch_size)

	return loader


def files_to_data(files):
	# preparation for the dataloader

	data_all = []

	for fn in files:
		data = A.load_data(fn)

		assert(np.all(~np.isnan(data['Data']))) # data should not have nan values

		data_all += [(d, y, t, fn) for (d, y, t) in zip(data['Data'], data['Y'], data['TimeStamps'])]

	data_seizure = [(d, y, t, fn) for (d, y, t, fn) in data_all if y == 1]
	data_nonseizure = [(d, y, t, fn) for (d, y, t, fn) in data_all if y == 0]
	
	return data_seizure, data_nonseizure


def get_training_data(data_set):
	"""
	Merge all seizure/non-seizure segments into one file to create a 'bag of segments'

	Parameters:
		data_set: string 

	Output:
		list with seizure segments
		list with non-seizure segments
	"""

	print('Preparing data...')
	
	files = [f'{x[0]}/{fn}' for x in os.walk(data_set) for fn in x[2]] 
	return files_to_data(files)


def patient_files(data_set):
	# files for each patient separately

	P = dict()

	for x in os.walk(f'{C.preprocessed_data_folder}/{data_set}/'): 
		for fn in x[2]:
			pid = fn.split('_')[0]
			
			P.setdefault(pid, [])
			P[pid] = P[pid] + [f'{x[0]}/{fn}']

	return P





