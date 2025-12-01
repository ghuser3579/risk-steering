'''
Pre-training the representation learner for the Imaging Only risk refinement
with PI-RADS constrained contrastive learning

The image representation learner is trained to transform bi-parametric MR images of the prostate into
low-dimensional representations such that these representations are indicative of the 
underlying risk of prostate cancer associated with these examinations.

PI-RADS contrained contrastive loss pushes representations of MRI exams with similar PI-RADS assessments 
closer to one another and pushes representations of dissimilar PI-RADS assessments away

Author LU
'''
 
import os, sys
import logging
import pathlib
 
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

curr_wd = os.getcwd()
sys.path.append(curr_wd) 

from config import pirads_constrained_pretrain_config as cfg
from utils.LayerDefinitions import initialize_representation_learner
from ContrastiveLoss import Constrained_Contrastive_Loss
from Data_generator import Baseline_Data_Generator
 
'''  
Instantiate image representation learner (RL)
Image RL is pretrained in steps - 
1) two contrast-specific CNN encoders for T2WI (axt2) and DWI (diff)
2) trained CNN encoders along with contrast aggregating transformer to generate subject-specific representation
'''

model_temp = initialize_representation_learner(cfg)


if cfg.contrast_type == 'biparametric':  # training the entire image RL
  model_ft = model_temp 
elif cfg.contrast_type == 'axt2':        # training the T2WI contrast-specific CNN encoder
  layer_name = 'contrast_learner_axt2'
  print(f'Creating contrast learner using {layer_name}')
  relevant_layer = model_temp.get_layer(layer_name)
  model_ft = tf.keras.Model(relevant_layer.inputs,relevant_layer.output)
else:                                    # training the DWI contrast-specific CNN encoder
  layer_name = 'contrast_learner_diff'
  print(f'Creating contrast learner using {layer_name}')
  relevant_layer = model_temp.get_layer(layer_name)
  model_ft = tf.keras.Model(relevant_layer.inputs,relevant_layer.output)

print(f'Instantiated representation learner for {cfg.contrast_type} contrast')

print('Model summary')
model_ft.summary()
 
cfg.hidden_dim = model_ft.output.shape[-1]


''' Instantiate PI-RADS guided contrastive learning loss'''
loss = Constrained_Contrastive_Loss(cfg)
custom_loss = loss.calc_pirads_constrained_contrastive_loss

 
''' Data generator'''
print('Loading baseline datagenerator')
traindatagen = Baseline_Data_Generator.DataLoader_Volume(cfg, pretrainMode=True)
valdatagen = Baseline_Data_Generator.DataLoader_Volume(cfg, pretrainMode=True, isTrain=False)
train_steps_per_epoch = traindatagen.num_samples
val_steps_per_epoch = valdatagen.num_samples
print(f'Num batches in train set: {train_steps_per_epoch}')
print(f'Num batches in val set: {val_steps_per_epoch}')

#%% SAVE/LOG LOCATIONS
# Create directories for saving log and checkpoints
# This string is used to create save directory
cfg.task_type = cfg.lr_mode + 'LR'+ str(cfg.lr) + '_epochs' + str(cfg.epochs) + 'tau' + str(cfg.temperature)  

cfg.save_dir = os.path.join(cfg.save_dir, cfg.dataset, cfg.task_type)
pathlib.Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
csv_path = cfg.save_dir + '/logs.csv'
print(f'Saving weights/csv files at {cfg.save_dir}')

print('Compiling model')
csv_path = cfg.save_dir + '/logs.csv'
modelsavepath = cfg.save_dir + '/chkpt'

# ADAMW optimizer with constant learning rate
optimizer = tfa.optimizers.AdamW(learning_rate=cfg.lr,
                                           weight_decay=cfg.weight_decay)
model_ft.compile(optimizer=optimizer, 
                loss=custom_loss, run_eagerly=True)
callbacks = [tf.keras.callbacks.ModelCheckpoint(modelsavepath,
                                          monitor='val_loss',
                                          verbose=1,
                                          save_best_only=True,
                                          save_weights_only=True),
            tf.keras.callbacks.CSVLogger(csv_path),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.5,
                                           patience=5)]

#%% MODEL TRAINING
# Train model
print('Starting training for epochs', cfg.epochs)
 
history = model_ft.fit(traindatagen,
                      steps_per_epoch=train_steps_per_epoch,
                      verbose=1,
                      epochs=cfg.epochs,
                      callbacks=callbacks,
                      validation_data=valdatagen,
                      validation_steps=val_steps_per_epoch,
                      initial_epoch=cfg.initial_epoch,
                      use_multiprocessing=False
                      )
# Save final weights
print('Saving the model weights at',cfg.save_dir)
model_ft.save_weights(os.path.join(f'{cfg.save_dir}/RL_model_weights.hdf5'))
