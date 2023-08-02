
# TAKEN FROM: https://github.com/gpleiss/temperature_scaling

import torch
from torch import nn, optim
from torch.nn import functional as F

import aux as A  
import dataloader as DL 


class ModelWithTemperature(torch.nn.Module):
	"""
	A thin decorator, which wraps a model with temperature scaling
	model (nn.Module):
		A classification neural network
		NB: Output of the neural network should be the classification logits,
			NOT the softmax (or log softmax)!
	"""
	def __init__(self, model):
		super().__init__()

		self.model = model
		self.model.eval() 

		self.temperature = nn.Parameter(torch.ones(1) * 1.5)

	def forward(self, input):
		logits = self.model(input)
		return self.temperature_scale(logits)

	def temperature_scale(self, logits):
		"""
		Perform temperature scaling on logits
		"""
		# Expand temperature to match the size of logits
		temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
		return logits / temperature

	# This function probably should live outside of this class, but whatever
	def set_temperature(self, device, valid_loader):
		"""
		Tune the tempearature of the model (using the validation set).
		We're going to set it to optimize NLL.
		valid_loader (DataLoader): validation set loader
		"""
		self.to(device)
		nll_criterion = nn.CrossEntropyLoss().to(device)

		# First: collect all the logits and labels for the validation set
		logits_list = []
		labels_list = []
		with torch.no_grad():
			for input, label in valid_loader:
				input = input.to(device)
				logits = self.model(input)
				logits_list.append(logits)
				labels_list.append(label)
			logits = torch.cat(logits_list).to(device)
			labels = torch.cat(labels_list).to(device)

		# Calculate NLL and ECE before temperature scaling
		before_temperature_nll = nll_criterion(logits, labels).item()
		print(f'Before temperature - NLL: {before_temperature_nll}')

		# Next: optimize the temperature w.r.t. NLL
		optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=500)

		def eval():
			optimizer.zero_grad()
			loss = nll_criterion(self.temperature_scale(logits), labels)
			loss.backward()
			return loss
		optimizer.step(eval)

		print('Optimal temperature: %.3f' % self.temperature.item())

		after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
		print(f'After temperature - NLL: {after_temperature_nll}')

		return self


def train(device, model, model_name, data, normalize = False, batch_size = 1024):

	model.to(device)

	loader_val =  DL.get_dataloader(data, balance = False, normalize = normalize, batch_size = batch_size, shuffle = True)

	model_temperature = ModelWithTemperature(model)
	model_temperature.set_temperature(device, loader_val)

	torch.save(model_temperature, model_name.replace('model_', 'model_final_'))

	return model_temperature

