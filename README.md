# Calibration of Seizure Detection Algorithms (SDAs)

## Requirements 
- Python >= 3.8
- PyTorch 

## Data 
- Adult data can be downloaded [here](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml#c_tusz) [1]. Load it into the `data/adult` folder.
- Neonatal data can be downloaded [here](https://zenodo.org/record/4940267#.Ybcah33P1hE) [2]. Load it into the `data/neonatal` folder.

## Preprocessing, training and evaluation
See [example](src/example_adult.py).

## References

[1] Shah, Vinit, et al. "The temple university hospital seizure detection corpus." Frontiers in neuroinformatics 12 (2018): 83.

[2] Stevenson, Nathan J., et al. "A dataset of neonatal EEG recordings with seizure annotations." Scientific data 6.1 (2019): 1-8.
