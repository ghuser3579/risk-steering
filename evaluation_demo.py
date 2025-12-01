'''
Sample script for evaluation

Given an accession number curr_accnum and relevant prior information available in './csv/imaging_evaluation_with_priors_demo.csv'
initial risk assessment and refinement is performed
Inputs - 
1. curr accession number in curr_accnum
2. csv file containing information about prior context in imaging_csv
3. data directory containing all MRI data pre-processed as h5 or nii or mha files available in data_directory './demo_data/'
4. number of prior MRI exams to use in num_timepoints_to_use

'''


import pandas as pd
import numpy as np
import sys
import os

curr_wd = os.getcwd()
sys.path.append(curr_wd) 

import tensorflow as tf

from data.data_processing import get_imaging_priors_demo

from config import risk_assm_config as RL_cfg
from config import temporal_learner_config as TL_cfg
from utils.ModelDefinitions import Context_aware_imaging_only_model


data_directory = './demo_data/'
imaging_csv = './csv/imaging_evaluation_with_priors_demo.csv'
checkpoint_wts = './pretrained_wts/imaging_chkpt'
 

''' Model Instantiation '''
ca_imaging_model = Context_aware_imaging_only_model(RL_cfg, TL_cfg)

''' Loading weights from checkpoint'''
ca_imaging_model.load_weights(checkpoint_wts)

''' Get current and prior information'''
# Load csv files with prior information
df = pd.read_csv(imaging_csv,
                    converters={'Curr_AccNum': pd.eval, 'Curr_Age': pd.eval, 
                                'Previous_AccNum': pd.eval, 'Previous_Age':pd.eval, 
                                'Curr_MaxPIRADS': pd.eval})


current_accnum = 1000044

# get prior and current MRI exams along with time intervals between exams
i_prior, ti_prior, i_recent, ti_recent =  get_imaging_priors_demo(current_accnum, df, data_directory, num_timepoints_to_use = 3)

''' Get initial and refined risk'''
initial_risk, refined_risk = ca_imaging_model.evaluate_risk([i_prior, ti_prior, i_recent, ti_recent])
 
print(f'PCa risk with current exam {initial_risk[0]}')
print(f'PCa risk with prior history {refined_risk[0]}')
