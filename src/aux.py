
import pickle


def save_data(file_name, data):
	pickle.dump(data, open(file_name, 'wb'))


def load_data(file_name):
	return pickle.load(open(file_name, 'rb'))


def get_fig_size(fig_width_pt, inches_per_pt, fig_scale, ratio):
	fig_width = fig_width_pt*inches_per_pt*fig_scale
	fig_height = fig_width*ratio               
	return [fig_width, fig_height]


def fig_pad(min_value, max_value, pecentage):
	d = abs(max_value - min_value)
	return d*pecentage