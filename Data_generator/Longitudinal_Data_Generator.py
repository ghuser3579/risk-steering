# -*- coding: utf-8 -*-
"""
Data generator for two timepoint prostate data with current and/or future labels
Uses new csv files that contain Age and 5 year diagnosis with 5 year mask information
Uses mask information to only retrieve datapoints for which labels exist
Returns labels and masks upto followup_yr.
"""
import tensorflow as tf
import numpy as np
import h5py
import threading

import os
import sys
import pandas as pd
from skimage.transform import resize

# Replace this path with parent folder location
sys.path.append('/Add parent directory/') 

from Augment_pretrain import AugmentObject  
from risk_utils import myCrop3D

class DataLoader_Volume(tf.keras.utils.Sequence):
    def __init__(self, cfg,
                isTrain=True,
                isLongitudinal=True,
                followup_yr=0
                ):    
        self.is_train = isTrain
        self.data_dir = cfg.data_dir 
        self.csv_dir = cfg.csv_dir        
        self.contrast_type = cfg.contrast_type
        self.isLongitudinal = isLongitudinal
        self.followup_yr = followup_yr
        print('Loading data from ', cfg.data_dir) 
        self.num_vols = self.get_meta_data()
        self.vol_indices = np.arange(self.num_vols)
        print('Number of training subjects:', self.num_vols)
        self.img_x = cfg.img_size_x
        self.img_y = cfg.img_size_y   
        self.img_z = cfg.img_size_z
        self.opShape = (cfg.img_size_x,cfg.img_size_y)
        self.batch_size = cfg.batch_size 
        self.num_samples = self.num_vols  // self.batch_size
        self.cfg = cfg
        self.axt2_bbox_shape = (cfg.axt2_x,cfg.axt2_y,cfg.axt2_z)
        self.diff_bbox_shape = (cfg.diff_x,cfg.diff_y,cfg.diff_z)
        self.set_aug_params()
        self.aug = AugmentObject(self.cfg)
        # random shuffling of volume indices
        np.random.shuffle(self.vol_indices)


    def set_aug_params(self):
        ''' Setting parameters for augmentation module'''
        self.cfg.max_delta = 0.5   # random brightness
        self.cfg.lower_cf = 0.5    # lower bound for contrast factor
        self.cfg.upper_cf = 1.5    # upper bound for contrast factor
        self.cfg.aug_seed = 1
        self.cfg.central_fraction = 0.95   # central crop parameter
        self.cfg.deform_axis = (0,1)   # in-plane elastic deformation
        self.cfg.deform_sigma = 2
    

    def get_meta_data(self):
        ''' Return data list from csv file based on training/validation set
        Each csv file has Curr Acc | Curr Age | Diagnosis 5yr | Mask 5yr | Previous Acc | Previous Age 
        '''
        if self.is_train:
            csv_file = 'Longitudinal_5yrdx_temporalaug_train.csv'
        else:
            csv_file = 'Longitudinal_5yrdx_temporalaug_val.csv'

        self.data_list = self.get_relevant_datalists(csv_file)
        return len(self.data_list['acc_num'])
    

    def get_relevant_datalists(self, csv_file):
        ''' Only generate lists for acc num and years for which data exists in the followup year'''

        df = pd.read_csv(os.path.join(self.csv_dir,csv_file),
                         converters={'Curr_AccNum': pd.eval, 'Curr_Age': pd.eval, 
                                     'Diagnosis_5year': pd.eval, 'Mask_5year': pd.eval,
                                     'Previous_AccNum': pd.eval, 'Previous_Age':pd.eval})
        # only read entries where Previous Accession Number is not null
        
        acc_num = np.asarray(df['Curr_AccNum'].values.tolist())
        age = np.asarray(df['Curr_Age'].values.tolist())
        prev_acc_num = np.asarray(df['Previous_AccNum'].values.tolist(),dtype=object)
        prev_age = np.asarray(df['Previous_Age'].values.tolist(),dtype=object)
        diagnosis_5yr  = np.asarray(df['Diagnosis_5year'].values.tolist())
        mask_5yr = np.asarray(df['Mask_5year'].values.tolist())
        # convert labels for bins [0,1,2,3,4,5] years into three temporal bins - current, in 2, in 5 
        diagnosis_3bins,mask_3bins = self.generate_label_per_bin(diagnosis_5yr, mask_5yr)
        print(f'Generating relevant data list from {csv_file}')
        data_list = {}
        data_list['acc_num'] = acc_num 
        data_list['prev_acc_num'] = prev_acc_num
        data_list['curr_age'] = age 
        data_list['diagnosis'] = diagnosis_3bins 
        data_list['mask'] = mask_3bins 
        data_list['prev_age'] = prev_age   
        return data_list
 
    
    def __len__(self):
        ''' Return length of samples in one training epoch'''
        return self.num_samples

    def get_len(self):
        ''' Return length of samples in one training epoch'''
        return self.num_samples

    def __shape__(self):
        data = self.__getitem__(0)
        return data.shape
    
    def on_epoch_end(self):
        '''Randomly shuffle order of accessing data at the end of every epoch'''
        np.random.shuffle(self.vol_indices)
 
    def __getitem__(self, idx):
        ''' Returns a dictionary corresponding to the batch idx'''
        with threading.Lock():
            '''Generate indices for the hdf5 files to load for the current batch'''
 
            curr_vol_idx = self.vol_indices[idx * self.batch_size : (idx+1) * self.batch_size]
            X_axt2    = np.zeros((self.batch_size, self.img_x, self.img_y, self.img_z, 1), dtype="float32")
            X_diff    = np.zeros((self.batch_size, self.img_x, self.img_y, self.img_z, 2), dtype="float32")
            diagnosis = np.zeros((self.batch_size,3), dtype='float32')
            mask = np.zeros((self.batch_size,3), dtype='float32')
            age_curr = np.zeros((self.batch_size,1)).astype('int32')

            if self.isLongitudinal:
                age_prev = np.zeros((self.batch_size,1)).astype('int32')
                X_axt2_prev = np.zeros((self.batch_size, self.img_x, self.img_y, self.img_z, 1), dtype="float32")
                X_diff_prev = np.zeros((self.batch_size, self.img_x, self.img_y, self.img_z, 2), dtype="float32")
                
            for jj in range(0, self.batch_size): 
                # current time point      
                filename = str(self.data_list['acc_num'][curr_vol_idx[jj]]) + '.h5' 
                X_axt2[jj], X_diff[jj]  = self.get_images_singletimepoint(filename) 
                diagnosis[jj] = self.data_list['diagnosis'][curr_vol_idx[jj]]
                mask[jj] = self.data_list['mask'][curr_vol_idx[jj]]
                curr_age = self.data_list['curr_age'][curr_vol_idx[jj]]
                # Average age in population dataset is 65. If this csv entry is not available, use the average age
                if curr_age == 0:
                    curr_age = 65
                age_curr[jj] = curr_age
                if self.isLongitudinal:
                    # randomly select a prior timepoint
                    prev_acc, prev_age = self.select_previous_time_point(curr_vol_idx[jj])
                    X_axt2_prev[jj], X_diff_prev[jj]  = self.get_images_singletimepoint(str(prev_acc) + '.h5') 
                    if prev_age == 0:
                        prev_age = 65
                    age_prev[jj] = prev_age
                
            labels = np.stack((diagnosis,mask),axis=-1)
            if self.isLongitudinal:
                # current time position index set to 1 (not 0)
                curr_time = np.ones((self.batch_size,1)).astype('int32')
                # prior time position in years elapsed between prior and current
                prior_time = age_curr - age_prev + curr_time
                prior_img = [X_axt2_prev, X_diff_prev] 
                curr_img = [X_axt2, X_diff] 
                data = (prior_img, prior_time, curr_img, curr_time)

                return data, labels
            else:  
                if self.contrast_type == 'axt2':
                    img = X_axt2
                elif self.contrast_type == 'diff':
                    img = X_diff
                else:
                    img = [X_axt2, X_diff]
                return img, labels
 
      
    def select_previous_time_point(self,idx):
        prev_acc_num_seq = self.data_list['prev_acc_num'][idx]
        prev_age_seq = self.data_list['prev_age'][idx]
        num_timepoints = len(prev_acc_num_seq)
        if self.is_train:
            time_to_select = np.random.choice(num_timepoints)
        else:
            time_to_select = -1   # Select the most recent timepoint
        prev_acc = prev_acc_num_seq[time_to_select]
        prev_age = prev_age_seq[time_to_select]
        return prev_acc, prev_age
        
    def read_hdf5_train(self, vol_idx):
        filename = str(self.data_list['acc_num'][vol_idx]) + '.h5'
        return self.get_images(filename)
    
    def min_max_norm(self,img):
        max_ = img.max()
        min_ = img.min()
        img = (img-min_)/(max_-min_)
        img = img.astype('float32')
        return img
    
    def get_images_singletimepoint(self, filename):
        ''' retrieve images corresponding to an accession number '''
        # Get t2w and diffusion volumes for a subject from h5 files    
        with h5py.File(self.data_dir + filename, 'r') as f:
            t2 = f['axt2'][:]
            adc = f['adc'][:]
            b1500 = f['b1500'][:]
        
        # Process t2w data
        t2_crop = self.min_max_norm(t2)
        if self.is_train:
            # only augment for training set
            t2_crop = self.crop_aug_bbox(t2)
            t2 = self.aug.augmentDatav2(t2_crop.squeeze())
        else:
            t2_crop = self.crop_bbox(t2)
            t2 = t2_crop.squeeze()
        axt2_img = np.expand_dims(t2, axis=-1)
        
        # Process diffusion data
        adc = self.min_max_norm(adc)
        b1500 = self.min_max_norm(b1500)

        if self.is_train:
            diff =  self.crop_aug_bbox_mc([adc,b1500])
            diff_img = self.aug.augmentDatav4(diff)
        else:
            adc_crop = self.crop_bbox(adc)
            b1500_crop = self.crop_bbox(b1500)
            diff_img = np.stack((adc_crop, b1500_crop), axis=-1)
        return axt2_img, diff_img
    
    def crop_aug_bbox(self, x):
        ''' Use 3D crop function to retain the center of 3D volume.
        '''
        orig_img_shape = x.shape 
        img_size_x = np.random.choice([224,240,256,272,284]) 
        img_size_z = np.random.choice([0,1,2])
        # crop slice dim only
        x = x[...,:orig_img_shape[2]-img_size_z]
        cropped_img = resize(x, (img_size_x,img_size_x,24))
        cropped_img = myCrop3D(cropped_img, (self.cfg.diff_x,self.cfg.diff_y))
        cropped_img = cropped_img[...,4:-4]
        return cropped_img
    
    def crop_aug_bbox_mc(self, arr_list):
        cropped_img = np.zeros((self.cfg.diff_x,self.cfg.diff_y,self.cfg.diff_z,2))
        img_size_x = np.random.choice([224,240,256,272,284]) 
        img_size_z = np.random.choice([0,1,2])
        for idx in range(len(arr_list)):
            temp= arr_list[idx]
            temp = temp[...,:temp.shape[2]-img_size_z]
            temp = resize(temp, (img_size_x,img_size_x,24))
            temp = myCrop3D(temp, (self.cfg.diff_x,self.cfg.diff_y))
            cropped_img[...,idx] = temp[...,4:-4]
        return cropped_img

    def crop_bbox(self,img):
        ''' Use 3D crop function to retain the center of 3D volume.
        Use this functionality for diffusion data since we do not have
        prostate masks/trained model for this data'''
        cropped_img = resize(img, (256,256,24))
        cropped_img = myCrop3D(cropped_img, (self.diff_bbox_shape[0],self.diff_bbox_shape[1]))
        cropped_img = cropped_img[...,4:-4]
        return cropped_img
    
    def generate_label_per_bin(self, temporal_labels, mask_labels, bin_indices=[2,5]):
        ''' General PCA labels and mask labels for temporal bins. for e.g., [0],[1-3] and [3-6]'''
        batch_size = temporal_labels.shape[0]
        new_temporal_labels = np.zeros((batch_size,len(bin_indices)+1)) 
        new_mask_labels = np.zeros((batch_size,len(bin_indices)+1)) 
        bin_init_idx = 0
        new_temporal_labels[:,bin_init_idx] = temporal_labels[:,bin_init_idx]
        new_mask_labels[:,bin_init_idx] = mask_labels[:,bin_init_idx]
        
        for batch_idx in range(batch_size):
            bin_init_idx = 1
            for bin_idx in range(len(bin_indices)):
                new_temporal_labels[batch_idx,bin_idx+1] = np.any(temporal_labels[batch_idx,bin_init_idx:bin_indices[bin_idx]+1])
                new_mask_labels[batch_idx,bin_idx+1] = np.any(mask_labels[batch_idx,bin_init_idx:bin_indices[bin_idx]+1])
                bin_init_idx = bin_indices[bin_idx]+1

        return new_temporal_labels, new_mask_labels
        

  

 
   
 
    
     
  
    
  
    