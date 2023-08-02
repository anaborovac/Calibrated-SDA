
import numpy as np
import sklearn.metrics as m
import torch
import scipy.special


def calculate_classification_metrics(target, predictions, probabilities):
	"""
	Calculation of classification metrics 

	Parameters:
		target - array with target values (0/1)
		predictions - array with predictions (0/1)
		probabilities - array with probabilities associated with predicitons (0-1)

	Output:
		AUC
		accuracy in percentages
		sensitivity in percentages
		specificity in percentages
	"""

	((tn, fp), (fn, tp)) = m.confusion_matrix(target, predictions, labels = [0, 1])

	auc = m.roc_auc_score(target, probabilities) if (1 in target) and (0 in target) else None
	acc = ((tp + tn) / (tp + tn + fp + fn)) * 100
	se = (tp / (tp + fn)) * 100 if 1 in target else None
	sp = (tn / (tn + fp)) * 100 if 0 in target else None

	return auc, acc, se, sp


def split_confidences_into_bins(target, predictions, confidences, n_bins):
	# modified from https://github.com/gpleiss/temperature_scaling

	bin_boundaries = np.linspace(0.5, 1, n_bins + 1) # lower bound = 0.5 since we have a binary classification problem
	bin_lowers = bin_boundaries[:-1]
	bin_uppers = bin_boundaries[1:]

	bin_lowers[0] = bin_lowers[0] - 0.01 # in case a confidence is exactly 0.5, so we do not skip it

	bins = [(target[np.logical_and(confidences > bl, confidences <= bu)], predictions[np.logical_and(confidences > bl, confidences <= bu)], confidences[np.logical_and(confidences > bl, confidences <= bu)]) 
				for bl, bu in zip(bin_lowers, bin_uppers)]

	bin_lowers[0] = bin_lowers[0] + 0.01

	return bins, bin_lowers, bin_uppers


def split_confidences_into_equally_sized_bins(target, predictions, confidences, n_bins):

	sort_confidences = np.argsort(confidences)

	target_split = np.array_split(target[sort_confidences], n_bins)
	predictions_split = np.array_split(predictions[sort_confidences], n_bins)
	confidences_split = np.array_split(confidences[sort_confidences], n_bins)

	bins = list(zip(target_split, predictions_split, confidences_split))

	bin_boundaries = [0.5] + [(confidences_split[i][-1] + confidences_split[i+1][0])/2 for i in range(n_bins-1)] + [1]
	bin_boundaries = np.array(bin_boundaries)

	bin_lowers = bin_boundaries[:-1]
	bin_uppers = bin_boundaries[1:]

	return bins, bin_lowers, bin_uppers


def expected_calibration_error(bins):

	ece = 0
	N = 0

	for (t, p, c) in bins:

		assert(len(t) == len(p) == len(c))

		n = len(c)

		if n == 0: continue

		acc = np.mean(t == p)
		conf = np.mean(c)

		ece += np.abs(acc - conf)*n
		N += n

	return ece/N * 100


def overconfidence_error(bins):

	oe = 0
	N = 0

	for (t, p, c) in bins:

		assert(len(t) == len(p) == len(c))

		n = len(c)

		if n == 0: continue

		acc = np.mean(t == p)
		conf = np.mean(c)

		oe += max([conf - acc, 0])*n
		N += n

	return oe/N * 100


def static_calibration_error(bins_0, bins_1):

	ece_0 = expected_calibration_error(bins_0)
	ece_1 = expected_calibration_error(bins_1)

	sce =  (ece_0 + ece_1) / 2

	return sce


def brier_score(target, probabilities):

	bs = m.brier_score_loss(target, probabilities)
	
	return bs


def negative_log_likelihood(target, probabilities):

	n = target.shape[0]

	t = np.zeros((n, 2))
	t[:, 0] = 1 - target
	t[:, 1] = target 

	prob = probabilities.copy()
	prob[prob == 0] = 1e-5

	nll = t * np.log(prob)
	nll = np.sum(nll) / n
	
	return -nll


def calculate_calibration_metrics(target, predictions, probabilities, n_bins = 5):
	"""
	Calculation of calibration metrics 

	Parameters:
		target - array of size N with target values (0/1)
		predictions - array of size N with predictions (0/1)
		probabilities - array of size N with probabilities associated with predicitons (0-1)
		n_bins - number of bins the confidences are split into

	Output:
		expectec calibration error in percentages
		overconfidence error in percentages
		static calibration error in percentages
		Brier score
		negative log likelihood
	"""

	confidences = np.copy(probabilities[:, 1])
	confidences[predictions == 0] = np.copy(probabilities[predictions == 0, 0])

	bins, _, _ = split_confidences_into_bins(target, predictions, confidences, n_bins)
	bins_equal, _, _ = split_confidences_into_equally_sized_bins(target, predictions, confidences, n_bins)

	bins_nonseizure, _, _ = split_confidences_into_bins(target[target == 0], predictions[target == 0], confidences[target == 0], n_bins)
	bins_seizure, _, _ = split_confidences_into_bins(target[target == 1], predictions[target == 1], confidences[target == 1], n_bins)

	ece = expected_calibration_error(bins)
	oe = overconfidence_error(bins)
	sce = None if (0 not in target) or (1 not in target) else static_calibration_error(bins_nonseizure, bins_seizure)

	bs = brier_score(target, probabilities[:, 1])
	nll = negative_log_likelihood(target, probabilities)

	return ece, oe, sce, bs, nll
