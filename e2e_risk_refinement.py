'''
End to End risk refinement with prior context
This framework consists of image RL, risk prediction model, a temporal learner, and a risk refinement model.
The image RL transforms bi-parametric MR images of the prostate into a 256-dimensional latent representation.
The risk prediction model uses this subject-specific latent representation to predict risk of prostate cancer
at the time of the visit and within 5-years of the visit.
The temporal learner takes in a sequence of representations corresponding to current and prior MRI visits of a subject 
and aggregates them into a change signal.
A risk refinement model takes this change signal and steers (upgrades or downgrades) the risk assessment from the risk prediction
model.

'''
 
import os, sys
import logging
import pathlib
 
sys.path.append('./risk_steering')
 
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
 
from config import risk_assm_config as RL_cfg
from config import temporal_learner_config as TL_cfg
from LayerDefinitions import initialize_representation_learner
from LayerDefinitions import initialize_risk_prediction_model
from LayerDefinitions import initialize_temporal_learner
from risk_utils import custom_AUC, MaskedWeightedCategoricalCrossEntropy


# Create E2E model for training all the models together
 
class e2e_model(tf.keras.Model):
    def __init__(self, RL_args, TL_args):
        super().__init__()
        self.RL = initialize_representation_learner(RL_args)  # representation learner with projection head
        self.temporal_learner = initialize_temporal_learner(TL_args, num_classes=TL_args.aux_dim)   # temporal learner
        self.risk_prediction_model = initialize_risk_prediction_model(RL_args,                    
                                                max_followup = RL_args.MAX_FOLLOWUP-1, 
                                                longitudinal_data_flag=False, 
                                                activation=False)  # risk prediction model 
        self.risk_refinement_model = initialize_risk_prediction_model(RL_args, 
                                                max_followup = RL_args.MAX_FOLLOWUP-1, 
                                                longitudinal_data_flag=True, 
                                                activation=False)  # risk refinement model
        self.BN = tf.keras.layers.BatchNormalization(name='batch_norm')
        self.representation_learner = tf.keras.Model(self.RL.inputs,self.RL.get_layer('SimpleAttnPoolv2').output) # representation learner
        
    def call(self, data):
        prior_img, prior_time, curr_img, curr_time = data  
        # current representation
        z_curr = self.representation_learner(curr_img)
        # prior representation
        z_prev = self.representation_learner(prior_img)
        prev_input = [z_prev, prior_time]
        curr_input = [z_curr, curr_time]
        # aggregated change signal
        z_agg = self.temporal_learner([prev_input,curr_input])
        z_agg = self.BN(z_agg)
        # initial risk assessment from current representation
        pred_hazard = self.risk_prediction_model(z_curr)
        # refined risk assessment from the change signal
        refined_hazard = self.risk_refinement_model(z_agg)
        # risk steering
        cumul_hazard = tf.keras.layers.Add()([pred_hazard,refined_hazard])
        # risk to probabilities
        cumul_prob = tf.keras.layers.Activation('sigmoid')(cumul_hazard)
        return cumul_prob

model_ft = e2e_model(RL_cfg, TL_cfg)
print('Instantiate subclassed Model')
 
#%% Data generator

from Data_generator import Longitudinal_Data_Generator

print('Loading longitudinal datagenerator')
traindatagen = Longitudinal_Data_Generator.DataLoader_Volume(RL_cfg, isLongitudinal=True)
valdatagen = Longitudinal_Data_Generator.DataLoader_Volume(RL_cfg, isLongitudinal=True, isTrain=False)
train_steps_per_epoch = traindatagen.num_samples
val_steps_per_epoch = valdatagen.num_samples
print(f'Num batches in train set: {train_steps_per_epoch}')
print(f'Num batches in val set: {val_steps_per_epoch}')

 
'''
Data generator returns data from current visit and a randomly sampled prior visit
(prior_img, prior_time, curr_img, curr_time) = val_data
curr_img = [T2WI, ADC+b1500] where T2WI - (batch size)x128x128x16x1 and ADC+b1500 - (batch size)x128x128x16x2
'''
val_data, val_label = valdatagen.__getitem__(0)

pred = model_ft.predict(val_data)


#%% Compile Model

from risk_utils import MaskedWeightedCategoricalCrossEntropy

auc_metrics = custom_AUC(followup_yr=TL_cfg.FOLLOWUP_YR)
custom_loss = MaskedWeightedCategoricalCrossEntropy(followup_yrs=TL_cfg.FOLLOWUP_YR, 
                                                class_wts=TL_cfg.CLASS_WTS,
                                                year_wts=[1.,1.,1.])

optimizer = tfa.optimizers.AdamW(learning_rate=TL_cfg.LEARNING_RATE, 
                                  weight_decay=TL_cfg.WEIGHT_DECAY,
                                  beta_1=0.9,
                                  beta_2=0.999)

model_ft.compile(loss=custom_loss, optimizer=optimizer)

#%%  Save directory

save_dir = os.path.join(RL_cfg.save_dir, 'ImagingOnlyRiskRefinement',  + str(TL_cfg.LEARNING_RATE) + '_wdecay' + str(TL_cfg.WEIGHT_DECAY) + '_epochs_' + str(TL_cfg.EPOCHS)+ '_followup_' + str(TL_cfg.FOLLOWUP_YR))
print(f'Saving model weights at {save_dir}')
 
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

csv_path = save_dir + '/logs.csv'
modelsavepath = save_dir + '/chkpt'


#%% Call backs
callbacks = [tf.keras.callbacks.ModelCheckpoint(modelsavepath,
                                              monitor='val_loss',
                                              verbose=1,
                                              save_best_only=True,
                                              save_weights_only=True),
            tf.keras.callbacks.CSVLogger(csv_path),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                 factor=0.5,
                                                 patience=10, 
                                                 verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=20,
                                              verbose=1)]
 


#%% Model training

print('Starting training for epochs', TL_cfg.EPOCHS)
 
history = model_ft.fit(traindatagen,
                      steps_per_epoch=train_steps_per_epoch,
                      verbose=1,
                      epochs=TL_cfg.EPOCHS,
                      callbacks=callbacks,
                      validation_data=valdatagen,
                      validation_steps=val_steps_per_epoch,
                      initial_epoch=TL_cfg.INITIAL_EPOCH,
                      use_multiprocessing=False
                      )


print('Save final weights')
 
# Save as hdf5 without optimizer states
model_ft.save_weights(os.path.join(f'{save_dir}/E2E_weights.hdf5'),
          overwrite=True,
          options=None) 
 