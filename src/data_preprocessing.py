
import numpy as np 
import mne
import scipy.signal
import os
import csv

import aux as A 
import constants as C


def signal_preprocessing(fs, data, low_cut, high_cut, fs_final):
	"""
	Preprocess EEG signals 

	Parameters:
		fs - sampling frequency
		data - array of size N x M, where N is the number of channels and M is the number of samples in the EEG signal
		low_cut - low-cut frequency used for filtering
		high_cut - high-cut frequency used for filtering
		fs_final - target sampling frequency

	Output:
		filtered and downsampled data
	"""

	first_fs = 250 # downsample first to 250 Hz to avoid problems with higher sampling frequencies (e.g. 1000 Hz)
	
	preprocessed = scipy.signal.resample_poly(data, first_fs, fs, axis = 1)

	sos = scipy.signal.butter(8, [low_cut / (first_fs / 2), high_cut / (first_fs / 2)], btype = 'bandpass', output = 'sos')
	preprocessed = scipy.signal.sosfiltfilt(sos, preprocessed, axis = 1)

	preprocessed = scipy.signal.resample_poly(preprocessed, fs_final, first_fs, axis = 1)

	return preprocessed


def data_preprocessing(file_name, file_name_preprocessed, montage, segment_duration = 16, segment_overlap = 12, low_cut = 0.5, high_cut = 30, fs_final = 62):
	"""
	Preprocess EEG recordings 

	Parameters:
		file_name - file name of an edf 
		file_name_preprocessed - file name where the preprocessed data is to be saved
		montage - desired montage, a touple of active and reference electrodes
		segment_duration - duration of EEG segment in seconds
		segment_overlap - overlap between two EEG segments in seconds
		low_cut - low-cut frequency used for filtering
		high_cut - high-cut frequency used for filtering
		fs_final - target sampling frequency
		
	Output:
		list of EEG segments of duration segment_duration and overlapping for segment_overlap seconds
		list of starting times of EEG segments (in seconds)
	"""

	(active, reference) = montage

	# read edf
	data = mne.io.read_raw_edf(file_name, infer_types = True)

	# extract needed information
	raw_data = data.get_data(units = {'eeg': 'uV'})
	channels = data.ch_names
	channels = [c.upper().replace('-LE', '-REF') for c in channels]
	fs = int(data.info['sfreq'])
	duration = data.n_times / fs

	# assigning montage to the recorded EEG data
	data_montage = np.zeros((len(active), int(duration)*fs))
	for i, (a, r) in enumerate(zip(active, reference)):
		if (f'{a}-REF' not in channels) or (f'{r}-REF' not in channels):
			raise Exception(f'Missing channel: {a}-{r}')
		data_montage[i] = raw_data[channels.index(f'{a}-REF')] - raw_data[channels.index(f'{r}-REF')]
	
	# preprocess the EEG signals (filter and downsample)
	data_preprocessed = signal_preprocessing(fs, data_montage, low_cut = low_cut, high_cut = high_cut, fs_final = fs_final)

	# cut the recording into short segments
	step_size = (segment_duration - segment_overlap) * fs_final
	final_point = data_preprocessed.shape[1] - segment_duration * fs_final

	data_segments = [data_preprocessed[:, start_point : (start_point + segment_duration * fs_final)] for start_point in range(0, final_point, step_size)]
	time_stamps = [start_point / fs_final for start_point in range(0, final_point, step_size)] # in seconds

	# save the preprocessed data
	A.save_data(file_name_preprocessed, {'Data': np.array(data_segments), 'TimeStamps': np.array(time_stamps), 'Duration': duration, 'Fs': fs})

	return data_segments, time_stamps


def get_seizure_intervals(file_name):
	"""
	Read csv_bi files and extract adult seizure information

	Parameters:
		file_name - file name of an csv_bi
		
	Output:
		array where each row is one seizure; the first column is equal to 1 if annotation is cofident and 2 otherwise, the second and the third columns are biginning and ending times in seconds
	"""

	seizures = []

	with open(file_name, 'rt') as file:
		data = csv.reader(file)

		# skip first 6 rows
		for _ in range(6): next(data)

		for [_, ts, te, e, c] in data:

			if e == 'bckg': continue
			assert(e == 'seiz')

			et = 2 if float(c) != 1 else 1 # not confident annotations
			
			seizures.append([et, float(ts), float(te)])
	
	return np.array(seizures)


def get_seizure_intervals_neonates(pid, file_names):
	"""
	Read csv_bi files and extract neonatal seizure information 

	Parameters:
		pid - patient ID (integer between 1 and 79)
		file_names - file names of annotations, i.e. annotations_2017_A, annotations_2017_B and annotations_2017_C
		
	Output:
		array where each row is one seizure; the first column is equal to 1 if all experts agreed on annotation and 2 if at least one expert annotated a seizure, the second and the third columns are biginning and ending times in seconds
	"""

	assert(pid >= 1 and pid <= 79)

	annotations = []
	for i, file_name in enumerate(file_names):
		seizure_ann = np.genfromtxt(file_name, delimiter = ',')
		ann = seizure_ann[:, seizure_ann[0] == pid].reshape(-1)

		ann = ann[1:] # ignore first element as it is pid

		if np.any(np.isnan(ann)): # remove nan values
			ann = ann[:min(np.where(np.isnan(ann))[0])] 

		annotations.append(ann)

	annotations = np.array(annotations)
	seizure_confidences = np.mean(annotations, axis = 0)

	seizures = []

	# consensus seizures
	seizure_index = np.where(seizure_confidences == 1)[0]

	if len(seizure_index) != 0:
		seizure_index_diff = np.diff(seizure_index)

		seizure_start = seizure_index[np.where(seizure_index_diff != 1)[0] + 1]
		seizure_start = np.append([seizure_index[0]], seizure_start)

		seizure_end = seizure_index[np.where(seizure_index_diff != 1)[0]]
		seizure_end = np.append(seizure_end, seizure_index[-1])

		seizures += [[1, s, e+1] for (s, e) in zip(seizure_start, seizure_end)]

	# seizure annotated by at least one annotator but not all of them
	part_seizure_index = np.where(np.logical_and(seizure_confidences != 1, seizure_confidences != 0))[0]

	if len(part_seizure_index) != 0:
		part_seizure_index_diff = np.diff(part_seizure_index)

		part_seizure_start = part_seizure_index[np.where(part_seizure_index_diff != 1)[0] + 1]
		part_seizure_start = np.append([part_seizure_index[0]], part_seizure_start)

		part_seizure_end = part_seizure_index[np.where(part_seizure_index_diff != 1)[0]]
		part_seizure_end = np.append(part_seizure_end, part_seizure_index[-1])

		seizures += [[2, s, e+1] for (s, e) in zip(part_seizure_start, part_seizure_end)]
	
	return np.array(seizures)


def label_segments(seizures, data_file_name, segment_duration = 16, bckg_overlap = True):
	"""
	Add labels to EEG segments 

	Parameters:
		seizures - array with seizure intervals, i.e. output of get_seizure_intervals and get_seizure_intervals functions
		data_file_name - file name where EEG segments are saved, i.e. output of data_preprocessing function
		segment_duration - duration of each EEG segment
		bckg_overlap - True if segments should overlap for non-seizure segments and False otherwise
		
	Output:
		dictionary with 
			Data - EEG segments 
			Y - labels; 1 if EEG segment is a seizure segment, 0 if it is non-seizure segment and 2 otherwise , e.g. if it is partly seizure, partly non-seizure or it is not confident/consensus annotation
			SeizureDuration - seizure durations for each EEG segment 
			TimeStamps - beginning of each EEG segment in seconds
			Seizures - seizure intervals
	"""

	assert(os.path.isfile(data_file_name))

	data_eeg = A.load_data(data_file_name)
	
	time_stamps, data_segments = data_eeg['TimeStamps'], data_eeg['Data']

	# sort EEG segments based on time
	sort_ts = np.argsort(time_stamps)
	time_stamps = time_stamps[sort_ts]
	data_segments = data_segments[sort_ts]
	
	y = np.zeros(len(time_stamps)) # labels
	seizure_duration = np.zeros(len(time_stamps))

	if len(seizures) != 0:
		for i, t_start in enumerate(time_stamps):
			t_end = t_start + segment_duration

			tmp_seizures = seizures[np.logical_and(t_start >= seizures[:, 1], t_end <= seizures[:, 2])] # check if the segment is "inside" of a seizure
			assert(tmp_seizures.shape[0] <= 1)
			y[i] = tmp_seizures[0][0] if tmp_seizures.shape[0] == 1 else 0
			seizure_duration[i] = tmp_seizures[0][2] - tmp_seizures[0][1] if tmp_seizures.shape[0] == 1 else 0

			tmp_seizures = seizures[np.logical_and(t_start < seizures[:, 2], t_end > seizures[:, 1])] # check if the segment overlaps with a seizure
			y[i] = 2 if tmp_seizures.shape[0] != 0 and y[i] == 0 else y[i]

	data_eeg['Y'] = y 
	data_eeg['SeizureDuration'] = seizure_duration
	data_eeg['Data'] = data_segments
	data_eeg['TimeStamps'] = time_stamps
	data_eeg['Seizures'] = seizures

	# remove overlapping non-seizure segments if desired (bckg_overlap = False)
	if (0 in y) and (not bckg_overlap):
		ts_0 = time_stamps[y == 0]
		ts_0_final = [ts_0[0]]
		for t in ts_0[1:]:
			if t - ts_0_final[-1] >= segment_duration:
				ts_0_final += [t]

		data_eeg['Y'] = y[np.logical_or(y != 0, np.isin(time_stamps, ts_0_final))]
		data_eeg['Data'] = data_segments[np.logical_or(y != 0, np.isin(time_stamps, ts_0_final))]
		data_eeg['TimeStamps'] = time_stamps[np.logical_or(y != 0, np.isin(time_stamps, ts_0_final))]
		data_eeg['SeizureDuration'] = seizure_duration[np.logical_or(y != 0, np.isin(time_stamps, ts_0_final))]

	A.save_data(data_file_name, data_eeg)

	return data_eeg


def preprocess_adult(data_subset_folder, preprocessed_subset_folder, bckg_overlap = False):
	"""
	Preprocess all recordings in a folder

	Parameters:
		data_subset_folder - folder with EEG recordings, e.g. data/adult/train
		preprocessed_subset_folder - folder where preprocessed recordings are to be saved, e.g. data_preprocessed/adult/train
	"""

	for x in os.walk(data_subset_folder):
		for fn in x[2]:

			if '01_tcp_ar' not in x[0]: continue # skip recordings that are not is tcp_ar montage

			if '.edf' in fn: # go through all .edf files
				print(fn)

				fn_pp = fn.replace('.edf', '.pt')
				fn_ann = fn.replace('.edf', '.csv_bi')
	
				data_preprocessing(f'{x[0]}/{fn}', f'{preprocessed_subset_folder}/{fn_pp}', (C.active_electrodes, C.reference_electrodes))
				seizures = get_seizure_intervals(f'{x[0]}/{fn_ann}')
				label_segments(seizures, f'{preprocessed_subset_folder}/{fn_pp}', bckg_overlap = bckg_overlap)


def preprocess_neonatal(data_subset_folder, preprocessed_subset_folder, bckg_overlap = False):
	"""
	Preprocess all neonatal recordings in a folder

	Parameters:
		data_subset_folder - folder with EEG recordings, e.g. data/adult/train
		preprocessed_subset_folder - folder where preprocessed recordings are to be saved, e.g. data_preprocessed/adult/train
	"""

	seizure_annotation_files = [ f'{data_subset_folder}/annotations_2017_A.csv'
							   , f'{data_subset_folder}/annotations_2017_B.csv'
							   , f'{data_subset_folder}/annotations_2017_C.csv']

	for x in os.walk(f'{data_subset_folder}/'):
		for fn in x[2]:

			if '.edf' in fn: # go through all .edf files
				print(fn)

				fn_pp = fn.replace('.edf', '.pt')
				pid = int(fn.replace('.edf', '').replace('eeg', ''))

				data_preprocessing(f'{x[0]}/{fn}', f'{preprocessed_subset_folder}/{fn_pp}', (C.active_electrodes_neonates, C.reference_electrodes_neonates))
				seizures = get_seizure_intervals_neonates(pid, seizure_annotation_files)
				label_segments(seizures, f'{preprocessed_subset_folder}/{fn_pp}', bckg_overlap = bckg_overlap)





							

				








