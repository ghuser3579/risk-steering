'''
E2E representation learner and risk prediction pipeline to predict high-risk of prostate cancer  
This framework takes bi-parametric prostate MR images and predicts current risk, risk in 2 years, and risk in 5 years
This does not include risk refinement module that use prior longitudinal data to refine current estimate of risk
 
'''


import os, sys
import logging
import pathlib

curr_wd = os.getcwd()
sys.path.append(curr_wd) 


# Suppressing TF message printouts
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
import tensorflow_addons as tfa

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


from config import risk_assm_config as risk_cfg
from utils.LayerDefinitions import initialize_representation_learner
from utils.LayerDefinitions import initialize_risk_prediction_model
from Data_generator import Baseline_Data_Generator
from Data_generator import Longitudinal_Data_Generator
from utils.risk_utils import custom_AUC, MaskedWeightedCategoricalCrossEntropy, WeightedCategoricalCrossEntropy


#%% Instantiate image representation learner model
'''
Model takes bi-parametric MR images and generates a latent representation
'''

print('Instantiating representation learner')
model_temp = initialize_representation_learner(risk_cfg)

if risk_cfg.LOAD_PRETR_WTS:
  print(f'Load pretrained weights for RL from {risk_cfg.RL_init_wts }')
  model_temp.load_weights(risk_cfg.RL_init_wts)

# discarding projection head from pretrained RL for downstream task (risk prediction)
representation_learner = tf.keras.Model(model_temp.inputs,model_temp.get_layer('SimpleAttnPoolv2').output)

print('RL Model summary')
representation_learner.summary()

#%% Instantiate risk prediction module

'''
Model takes in a latent representation and predicts risk at three timepoints
Channel 0 - risk at the time of the visit
Channel 1 - risk within 2 years of the visit
Channel 2 - risk within 5 years of the visit
'''
risk_prediction_model = initialize_risk_prediction_model(risk_cfg, 
                                            max_followup = risk_cfg.MAX_FOLLOWUP-1, 
                                            longitudinal_data_flag=False, 
                                            activation=True)

print('Risk prediction Model summary')
risk_prediction_model.summary()
 

#%% Create E2E model for training RL and RP together

def create_e2e_model():
  inputs = representation_learner.inputs
  hidden = representation_learner(inputs)
  cumul_prob = risk_prediction_model(hidden)
  model = tf.keras.Model(inputs,cumul_prob)
  return model

model_ft = create_e2e_model()
print('E2E risk pred model summary')
model_ft.summary()

optimizer = tfa.optimizers.AdamW(learning_rate=risk_cfg.LEARNING_RATE, 
                                  weight_decay=risk_cfg.WEIGHT_DECAY,
                                  beta_1=0.9,
                                  beta_2=0.999)


'''
Choose an appropriate data generator and loss function based on 
which layers (i.e. Channel) of the risk prediction module are being trained.
'''
if risk_cfg.FOLLOWUP_YR == 0:   # we are learning to predict baseline risk i.e., risk at the time of imaging
  print('Loading baseline datagenerator')
  traindatagen = Baseline_Data_Generator.DataLoader_Volume(risk_cfg, pretrainMode=False)
  valdatagen = Baseline_Data_Generator.DataLoader_Volume(risk_cfg, pretrainMode=False, isTrain=False)
  train_steps_per_epoch = traindatagen.num_samples
  val_steps_per_epoch = valdatagen.num_samples
  print(f'Num batches in train set: {train_steps_per_epoch}')
  print(f'Num batches in val set: {val_steps_per_epoch}')

  auc_metrics = custom_AUC(followup_yr=risk_cfg.FOLLOWUP_YR)
  custom_loss = WeightedCategoricalCrossEntropy(followup_yr=risk_cfg.FOLLOWUP_YR,
                                                 class_wts=risk_cfg.CLASS_WTS)
else:                         # we are learning to predict future risks
  print('Loading longitudinal datagenerator')
  traindatagen = Longitudinal_Data_Generator.DataLoader_Volume(risk_cfg, isLongitudinal=False)
  valdatagen = Longitudinal_Data_Generator.DataLoader_Volume(risk_cfg, isLongitudinal=False, isTrain=False)
  train_steps_per_epoch = traindatagen.num_samples
  val_steps_per_epoch = valdatagen.num_samples
  print(f'Num batches in train set: {train_steps_per_epoch}')
  print(f'Num batches in val set: {val_steps_per_epoch}')
  custom_loss = MaskedWeightedCategoricalCrossEntropy(followup_yrs=risk_cfg.FOLLOWUP_YR, 
                                                  class_wts=risk_cfg.CLASS_WTS,
                                                  year_wts=[1.,1.,1.])



''' check if previous checkpoints exist and load the weights'''

if risk_cfg.FOLLOWUP_YR >0: 
  previous_chkpt = os.path.join(risk_cfg.save_dir, 'RP_train_'+ str(risk_cfg.LEARNING_RATE) + '_wdecay' + str(risk_cfg.WEIGHT_DECAY) + '_epochs_' + str(risk_cfg.EPOCHS)+ '_followup_' + str(risk_cfg.FOLLOWUP_YR-1) + '/chkpt')
  if previous_chkpt:
    model_ft.load_weights(previous_chkpt)
    print(f'Loaded previous checkpoint weights from {previous_chkpt}')

# Compile the model
if risk_cfg.FOLLOWUP_YR == 0:
  model_ft.compile(loss=custom_loss,metrics=auc_metrics, optimizer=optimizer)
else:
  model_ft.compile(loss=custom_loss, optimizer=optimizer)


# Create directories for saving checkpoint and final model weights
save_dir = os.path.join(risk_cfg.save_dir, 'RP_train_'+ str(risk_cfg.LEARNING_RATE) + '_wdecay' + str(risk_cfg.WEIGHT_DECAY) + '_epochs_' + str(risk_cfg.EPOCHS)+ '_followup_' + str(risk_cfg.FOLLOWUP_YR))
print(f'Saving model weights at {save_dir}')
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

csv_path = save_dir + '/logs.csv'
modelsavepath = save_dir + '/chkpt'

callbacks = [tf.keras.callbacks.ModelCheckpoint(modelsavepath,
                                              monitor='val_loss',
                                              verbose=1,
                                              save_best_only=True,
                                              save_weights_only=True),
            tf.keras.callbacks.CSVLogger(csv_path),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                 factor=0.5,
                                                 patience=20, 
                                                 verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=10,
                                              verbose=1)]
 
print('Starting training for epochs', risk_cfg.EPOCHS)
 
history = model_ft.fit(traindatagen,
                      steps_per_epoch=train_steps_per_epoch,
                      verbose=1,
                      epochs=risk_cfg.EPOCHS,
                      callbacks=callbacks,
                      validation_data=valdatagen,
                      validation_steps=val_steps_per_epoch,
                      initial_epoch=risk_cfg.INITIAL_EPOCH,
                      use_multiprocessing=False
                      )

print('Save final weights')
# Save as hdf5 without optimizer states
model_ft.save_weights(os.path.join(f'{save_dir}/E2E_weights.hdf5'),
          overwrite=True,
          options=None) 
 
# Also save the individual image representation learner and risk prediction weights
model_ft.get_layer('imageRL').save_weights(os.path.join(f'{save_dir}/representation_learner_weights.hdf5'))
model_ft.get_layer('risk_pred_model').save_weights(os.path.join(f'{save_dir}/risk_prediction_weights.hdf5'))