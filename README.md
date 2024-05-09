# Calibration of Seizure Detection Algorithms (SDAs)

This repository contains code for performing experiments described in

Borovac Ana, Agustsson David H., Runarsson Thomas P., and Gudmundsson Steinn. "Calibration Methods for Automatic Seizure Detection Algorithms." Machine Learning Applications in Medicine and Biology. Cham: Springer Nature Switzerland, 2024. 65-85.

## Requirements 
The code was tested using:
- Python 3.9.7
- PyTorch 1.10.0
- NumPy 1.20.3
- MNE 0.24.1
- SciPy 1.8.0
- scikit-learn 1.2.2

## Data 
- Adult data can be downloaded [here](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml#c_tusz) [1]. Load it into the `data/adult` folder.
- Neonatal data can be downloaded [here](https://zenodo.org/record/4940267#.Ybcah33P1hE) [2]. Load it into the `data/neonatal` folder.

## Preprocessing, training and evaluation
See [example](src/example_adult.py).

## References

[1] Shah Vinit, et al. "The temple university hospital seizure detection corpus." Frontiers in neuroinformatics 12 (2018): 83.

[2] Stevenson Nathan J., et al. "A dataset of neonatal EEG recordings with seizure annotations." Scientific data 6.1 (2019): 1-8.
