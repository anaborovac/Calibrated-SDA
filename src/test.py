
import numpy as np
import torch
import random

from temperature import ModelWithTemperature

import dataloader as DL
import constants as C
import aux as A
import metrics as M


@torch.no_grad()
def get_predictions(device, model, dataloader, dropout):
	"""
	Get predictions for data 

	Parameters:
		device: e.g., 'cuda' or cpu'
		model: model to be used to make predictions
		dataloader: data to be predicted
		dropout: True if dropout is enabled otherwise False

	Output:
		array with target labels
		array with logit values
		array with softmax outputs
	"""

	model.eval()
	model.to(device)

	# TODO make this more generic
	if dropout:
		model.dropout1.train()
		model.dropout5.train()

	logit, prob, target = [], [], []

	for data, labels in dataloader: 

		p = model.forward(data.to(device)).cpu()

		logit += p.tolist()
		prob += torch.softmax(p, dim = 1).tolist()
		target += labels.tolist()

	return np.array(target), np.array(logit), np.array(prob)


@torch.no_grad()
def get_predictions_ensemble(device, models, dataloader, dropout):
	"""
	Get predictions for data (multiple models)

	Parameters:
		device: e.g., 'cuda' or cpu'
		model: m models to be used to make predictions
		dataloader: data to be predicted
		dropout: True if dropout is enabled otherwise False

	Output:
		array with target labels
		array with m x logit values
		array with m x softmax outputs
	"""

	m = len(models)

	for model in models:
		model.eval()
		model.to(device)

		# TODO make this more generic
		if dropout:
			model.dropout1.train()
			model.dropout5.train()

	logit, prob, target = [[]]*m, [[]]*m, []

	for data, labels in dataloader: 

		data = data.to(device)

		for i in range(m):

			p = models[i].forward(data).cpu()

			logit[i] = logit[i] + p.tolist()
			prob[i] = prob[i] + torch.softmax(p, dim = 1).tolist()

		target += labels.tolist()

	return np.array(target), np.array(logit), np.array(prob)


def get_per_patient_predictions(device, models, data_set, normalize = False, batch_size = 1024, dropout = False):
	"""
	Get predictions for each patient in a data set

	Parameters:
		device: e.g., 'cuda' or cpu'
		models: m models to be used to make predictions
		data_set: path of the data set folder
		normalize: True is the input signals are normalized such that std = 1 and mean = 0, False otherwise
		batch_size: batch size
		dropout: True if dropout is enabled otherwise False

	Output:
		dictionary with patient ids as keys and predictions as values
	"""

	patient_files = DL.patient_files(data_set)
	
	predicitons = dict()
	for pid, files in patient_files.items():

		patient_data = DL.files_to_data(files, split_seizure = True)
		patient_dataloader = DL.get_dataloader(data, balance = False, normalize = normalize, shuffle = False, batch_size = batch_size)

		predicitons[pid] = get_predictions_ensemble(device, models, patient_dataloader, dropout = dropout)

	return predicitons











			
		


		




