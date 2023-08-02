
import matplotlib.pyplot as plt

import aux as A
import constants as C


def plot_train_val_loss(train_loss, val_loss):

	fig = plt.figure(figsize = A.get_fig_size(C.fig_width_pt, C.inches_per_pt, C.fig_scale, 1))
	ax = fig.add_subplot(111)

	ax.plot(train_loss, color = 'k', label = 'Train')
	ax.plot(val_loss, color = 'k', linestyle = '--', label = 'Validation')

	ax.legend()

	ax.set_xlabel('Epoch')
	ax.set_ylabel('Loss')

	ax.xaxis.set_major_formatter(C.formatter)
	ax.yaxis.set_major_formatter(C.formatter)

	fig.tight_layout()
	plt.show()
