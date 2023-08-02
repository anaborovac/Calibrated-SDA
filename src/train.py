
import torch
import  numpy as np 

from nsda import NSDA 
from nsda_dropout import NSDA_DROPOUT
import dataloader as DL
import metrics as M
import test as T


def mixup_data(device, x, y, alpha):
	# TAKEN FROM: https://github.com/facebookresearch/mixup-cifar10

	lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

	batch_size = x.size()[0]
	index = torch.randperm(batch_size, device = device)

	mixed_x = lam * x + (1 - lam) * x[index, :]
	y_a, y_b = y, y[index]

	return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
	# TAKEN FROM: https://github.com/facebookresearch/mixup-cifar10
	return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(device, model_name, data_train, data_val, epochs = 50, lr = 0.001, step_size = 20, gamma = 0.5, batch_size = 256, normalize = False, dropout = False, mixup = False, mixup_alpha = 0.3):
	"""
	Train an SDA 

	Parameters:
		device: e.g., 'cuda' or 'cpu'
		model_name: name of a model 
		data_train: tuple with list of training seizure segmnets and list of training non-seizure segments
		data_val: tuple with list of validation seizure segmnets and list of validation non-seizure segments
		epochs: number of epochs
		lr: learning rate of Adam optimizer
		step_size: learning rate step size 
		gamma: a number with which the learning rate is multiplied every step size
		batch_size: batch size
		normalize: True is the input signals are normalized such that std = 1 and mean = 0, False otherwise
		dropout: True if SDA with dropout is used, False otherwise
		mixup: True if SDA with mixup is used, False otherwise
		mixup_alpha: a float between 0 and 1

	Output:
		trained SDA
		list of training losses in each epoch
		list of validation losses in each epoch
		list of validation classification and calibration metrics in each epoch
	"""

	loader_train =  DL.get_dataloader(data_train, balance = True, normalize = normalize, batch_size = batch_size, shuffle = True)
	loader_val =  DL.get_dataloader(data_val, balance = False, normalize = normalize, batch_size = batch_size, shuffle = False)

	model = NSDA_DROPOUT() if dropout else NSDA()
	model.to(device)

	opt = torch.optim.Adam(model.parameters(), lr = lr)
	loss_func = torch.nn.CrossEntropyLoss(reduction = 'mean') 
	scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size = step_size, gamma = gamma) 

	loss_train_all, loss_val_all, metrics_val_all = [], [], []

	for epoch in range(epochs):
		print(epoch)

		loss_train = []
		n_train = 0

		model.train()
		for data, labels in loader_train: 

			opt.zero_grad()
		
			data, labels = data.to(device), labels.to(device) 
			if mixup:
				data, labels_a, labels_b, lam = mixup_data(device, data, labels, mixup_alpha) 

			pred = model.forward(data)

			if mixup:
				loss = mixup_criterion(loss_func, pred, labels_a, labels_b, lam) 
			else:
				loss = loss_func(pred, labels)

			n = data.shape[0]
			loss_train.append(loss.item()*n)
			n_train += n

			loss.backward()
			opt.step()

		scheduler.step()

		loss_train = np.sum(loss_train) / n_train

		# validation
		labels_val, logits_val, prob_val = T.get_predictions(device, model, loader_val, dropout = False)
		pred_val = np.argmax(prob_val, axis = 1)

		auc_val, acc_val, se_val, sp_val = M.calculate_classification_metrics(labels_val, pred_val, prob_val[:, 1])
		ece_val, oe_val, sce_val, bs_val, nll_val = M.calculate_calibration_metrics(labels_val, pred_val, prob_val, n_bins = 5)
		loss_val = loss_func(torch.tensor(logits_val), torch.tensor(labels_val).long()).item() 

		print(f'Training loss: {loss_train}')
		print(f'Val loss: {loss_val}')
		print(f'Val auc: {auc_val}, Val acc: {acc_val}, Val se: {se_val}, Val sp: {sp_val}')
		print(f'Val ECE: {ece_val}, Val OE: {oe_val}, Val SCE: {sce_val}, Val BS: {bs_val}, Val NLL: {nll_val}')

		loss_train_all.append(loss_train)
		loss_val_all.append(loss_val)
		metrics_val_all.append((auc_val, acc_val, se_val, sp_val, ece_val, oe_val, sce_val, bs_val, nll_val))

		torch.save(model, model_name.replace('model_', f'model_epoch_{epoch}_'))
	
	torch.save(model, model_name.replace('model_', 'model_final_'))

	return model, loss_train_all, loss_val_all, metrics_val_all








	
	