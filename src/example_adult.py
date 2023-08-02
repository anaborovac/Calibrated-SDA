
import torch

import constants as C
import data_preprocessing as DP 
import dataloader as DL
import train as T
import visualize as V 
import analysis as A
import temperature as TEMP
import test as TEST


# 1) PREPROCESS THE DATA

DP.preprocess_adult(f'{C.raw_data_folder}/adult/train', f'{C.preprocessed_data_folder}/adult/train')
DP.preprocess_adult(f'{C.raw_data_folder}/adult/dev', f'{C.preprocessed_data_folder}/adult/dev')
DP.preprocess_adult(f'{C.raw_data_folder}/adult/eval', f'{C.preprocessed_data_folder}/adult/eval')


# 2) PREPARE DATA FOR THE TRAINING

adult_train_seizure, adult_train_nonseizure = DL.get_training_data(f'{C.preprocessed_data_folder}/adult/train')
adult_val_seizure, adult_val_nonseizure = DL.get_training_data(f'{C.preprocessed_data_folder}/adult/dev')


# 3) TRAIN THE SDA

device = 'cuda' if torch.cuda.is_available() else 'cpu'
run_id = f'adult_test'

# 3.1) UNCALIBRATED SDA 

model_name = f'{C.models_folder}/model_{run_id}.pt'
model, train_loss, val_loss, _ = T.train(device, model_name, (adult_train_seizure, adult_train_nonseizure), (adult_val_seizure, adult_val_nonseizure), epochs = 1)
V.plot_train_val_loss(train_loss, val_loss)

# 3.2) SDA WITH TEMPERATURE SCALING

model_name_temperature = f'{C.models_folder}/model_{run_id}_temperature.pt'
model_temperature = TEMP.train(device, model, model_name_temperature, (adult_val_seizure, adult_val_nonseizure))

# 3.3) ENSEMBLE OF (3) SDAs

model_name_1 = f'{C.models_folder}/model_{run_id}_1.pt'
model_1, train_loss_1, val_loss_1, _ = T.train(device, model_name_1, (adult_train_seizure, adult_train_nonseizure), (adult_val_seizure, adult_val_nonseizure), epochs = 1)
V.plot_train_val_loss(train_loss_1, val_loss_1)

model_name_2 = f'{C.models_folder}/model_{run_id}_2.pt'
model_2, train_loss_2, val_loss_2, _ = T.train(device, model_name, (adult_train_seizure, adult_train_nonseizure), (adult_val_seizure, adult_val_nonseizure), epochs = 1)
V.plot_train_val_loss(train_loss_2, val_loss_2)

# 3.4) SDA WITH DROPOUT

model_name_dropout = f'{C.models_folder}/model_{run_id}_dropout.pt'
model_dropout, train_loss_dropout, val_loss_dropout, _ = T.train(device, model_name_dropout, (adult_train_seizure, adult_train_nonseizure), (adult_val_seizure, adult_val_nonseizure), dropout = True, epochs = 1)
V.plot_train_val_loss(train_loss_dropout, val_loss_dropout)

# 3.5) SDA WITH MIXUP

model_name_mixup = f'{C.models_folder}/model_{run_id}_mixup.pt'
model_mixup, train_loss_mixup, val_loss_mixup, _ = T.train(device, model_name_mixup, (adult_train_seizure, adult_train_nonseizure), (adult_val_seizure, adult_val_nonseizure), mixup = True, epochs = 1)
V.plot_train_val_loss(train_loss_mixup, val_loss_mixup)


# 4) TEST THE SDA

adult_eval_dataset = f'{C.preprocessed_data_folder}/data/eval' 

predictions = TEST.get_per_patient_predictions(device, [model], adult_eval_dataset)
predictions_temperature = TEST.get_per_patient_predictions(device, [model_temperature], adult_eval_dataset)
predictions_ensemble = TEST.get_per_patient_predictions(device, [model, model_1, model_2], adult_eval_dataset)
predictions_dropout = TEST.get_per_patient_predictions(device, [model_dropout]*10, adult_eval_dataset, dropout = True)
predictions_mixup = TEST.get_per_patient_predictions(device, [model_mixup], adult_eval_dataset)


# 5) CALCULATE METRICS

# 5.1) CALCULATE PATIENT-BASED METRICS

A.patient_based_metrics_ensemble(predictions)
A.patient_based_metrics_ensemble(predictions_temperature)
A.patient_based_metrics_ensemble(predictions_ensemble)
A.patient_based_metrics_ensemble(predictions_dropout)
A.patient_based_metrics_ensemble(predictions_mixup)

# 5.2) CALCULATE SEGMENT-BASED METRICS

A.segment_based_metrics_ensemble(predictions)
A.segment_based_metrics_ensemble(predictions_temperature)
A.segment_based_metrics_ensemble(predictions_ensemble)
A.segment_based_metrics_ensemble(predictions_dropout)
A.segment_based_metrics_ensemble(predictions_mixup)

