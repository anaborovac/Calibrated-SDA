
import numpy as np  

import metrics as M 


def patient_based_metrics_ensemble(T, model_id = None): 
	"""
	Calculation classification metrics for each patient saperately

	Parameters:
		T: dictionary with per-patient predictions
		model_id: int in case a certain model should be used instead of the whole ensemble

	Ouput:
		dictionary with classification metrics for each patient
	"""

	met = dict()
	met['AUC'], met['ACC'], met['SE'], met['SP'] = [], [], [], []

	seizure_patients = []

	for pid, (t, _, p) in T.items():

		assert(0 in t) # verify if there is at least one non-seizure segment
		if (1 not in t): continue # skip patients without seizures

		probabilities_ensemble = np.mean(p, axis = 0) if model_id is None else p[model_id] # average across ensemble
		predictions_ensemble = np.argmax(probabilities_ensemble, axis = 1)

		auc, acc, se, sp = M.calculate_classification_metrics(t, predictions_ensemble, probabilities_ensemble[:, 1])

		met['AUC'] = met['AUC'] + [auc]
		met['ACC'] = met['ACC'] + [acc]
		met['SE'] = met['SE'] + [se]
		met['SP'] = met['SP'] + [sp]

		seizure_patients += [pid]

	for m, v in met.items():
		met[m] = np.mean(v)
		print(f'{m}: {met[m]:.2f} ({np.std(v):.2f})')

	print(f'Number of patients with seizures: {len(seizure_patients)}')

	return met


def segment_based_metrics_ensemble(T, model_id = None):
	"""
	Calculation of classification and calibration metrics on all available EEG segments

	Parameters:
		T: dictionary with per-patient predictions
		model_id: int in case a certain model should be used instead of the whole ensemble

	Ouput:
		dictionary with classification and calibration metrics
	"""


	target, predictions, probabilities = np.empty((0,), int), np.empty((0,), int), np.empty((0, 2), float)
	for pid, (t, _, p) in T.items():

		if len(t) == 0: continue

		probabilities_ensemble = np.mean(p, axis = 0) if model_id is None else p[model_id] # average across ensemble
		predictions_ensemble = np.argmax(probabilities_ensemble, axis = 1)

		target = np.append(target, t, axis = 0)
		predictions = np.append(predictions, predictions_ensemble, axis = 0)
		probabilities = np.append(probabilities, probabilities_ensemble, axis = 0)

	met = dict()
	met['ECE'], met['OE'], met['SCE'], met['BS'], met['NLL'] = M.calculate_calibration_metrics(target, predictions, probabilities, n_bins = 5)
	met['AUC'], met['ACC'], met['SE'], met['SP'] = M.calculate_classification_metrics(target, predictions, probabilities[:, 1])

	for m, v in met.items():
		print(f'{m}: {v:.2f}')

	return met