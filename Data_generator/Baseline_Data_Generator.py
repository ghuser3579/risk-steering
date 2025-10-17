# -*- coding: utf-8 -*-
"""
Baseline Data Generator for training model
1) Representation learner when pretrainMode = True
2) Baseline layer of risk assessment module when pretrainModel=False
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
sys.path.append('/Add/parent/directory') 

from Augment_pretrain import AugmentObject  
from risk_utils import myCrop3D
 
class DataLoader_Volume(tf.keras.utils.Sequence):
    def __init__(self, 
                cfg,            # config file
                isTrain=True,   # True: training data gen, False: validation data gen
                pretrainMode=False,  # True: returns PI-RADS labels, False: returns binary labels PI-RADS>pirads_cutoff = 1
                pirads_cutoff = 3,
                normalization='minmax'): 
        
        self.is_train = isTrain
        self.normalization = normalization
        self.pretrainMode = pretrainMode
        self.pirads_cutoff = pirads_cutoff
        self.data_dir = cfg.data_dir      # location of h5 files
        self.csv_dir = cfg.csv_dir
        self.contrast_type = cfg.contrast_type   # requested output from datagen, options: 'axt2', 'diff', 'biparametric' 
        print('Loading data from ', cfg.data_dir)
        self.num_vols = self.get_pirads_meta_data_train()
        self.vol_indices = np.arange(self.num_vols)
        print('Number of training subjects:', self.num_vols)
        self.img_x = cfg.img_size_x
        self.img_y = cfg.img_size_y   
        self.img_z = cfg.img_size_z
        self.opShape = (cfg.img_size_x,cfg.img_size_y)
        self.bbox_shape = (cfg.img_size_x,cfg.img_size_y,cfg.img_size_z)  # bbox for crop function
        self.batch_size = cfg.batch_size 
        self.num_samples = self.num_vols  // self.batch_size 
        self.cfg = cfg
        self.set_aug_params()
        self.aug = AugmentObject(self.cfg)  
        # random shuffling of volume indices
        np.random.shuffle(self.vol_indices)
        
    def set_aug_params(self):
        ''' Setting parameters for augmentation module'''
        self.cfg.max_delta = 0.1   # random brightness
        self.cfg.lower_cf = 0.5    # lower bound for contrast factor
        self.cfg.upper_cf = 1.5    # upper bound for contrast factor
        self.cfg.aug_seed = 1
        self.cfg.central_fraction = 0.95   # central crop parameter
        self.cfg.deform_axis = (0,1)   # in-plane elastic deformation
        self.cfg.deform_sigma = 2
        
    def get_pirads_meta_data_train(self):
        '''
        These are csv files containing patients with no longitudinal follow-up, separated by corresponding max PI-RADS assessments
        '''
        acc1 = self.get_data_list('Single_timepoint_pirads_1_20240718.csv')
        acc2 = self.get_data_list('Single_timepoint_pirads_2_20240718.csv')
        acc3 = self.get_data_list('Single_timepoint_pirads_3_20240718.csv')
        acc4 = self.get_data_list('Single_timepoint_pirads_4_20240718.csv')
        acc5 = self.get_data_list('Single_timepoint_pirads_5_20240718.csv')
        # Combine all data and shuffle
        if self.pretrainMode:
            # exclude PI-RADS 3
            self.pirads_list = np.concatenate((acc1,acc2,acc4,acc5))
        else:
            # include PI-RADS 3
            self.pirads_list = np.concatenate((acc1,acc2,acc3,acc4,acc5))
        num_volumes =  len(self.pirads_list)
        return num_volumes

    def get_data_list(self,csv_file):
        ''' Load csv files with pandas and retrieve accession numbers'''
        df = pd.read_csv(os.path.join(self.csv_dir,csv_file),converters={'AccessionNumberList': pd.eval})
        temp = df['AccessionNumberList'].values.tolist()
        acc_num = []
        for idx in range(len(temp)):
            acc_num.append(temp[idx][0])
        acc_num = np.asarray(acc_num)
        np.random.seed(seed=256)
        shuffle_idx = np.arange(len(acc_num))
        np.random.shuffle(shuffle_idx)
        acc_num = acc_num[shuffle_idx]
        # Equal number of subjects from each group are set in validation set
        if self.is_train:
            acc_num = acc_num[:-100]
        else:
            acc_num = acc_num[-100:]
        return acc_num
    
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
        'Reshuffle the training set at the end of each epoch'
        self.vol_indices = np.arange(self.num_vols)
        np.random.shuffle(self.vol_indices)
 
    def __getitem__(self, idx):
        ''' Returns a dictionary corresponding to the batch idx'''
        with threading.Lock():
            '''Generate indices for the hdf5 files to load for the current batch'''
            curr_vol_idx = self.vol_indices[idx * self.batch_size : (idx+1) * self.batch_size]
            # placeholders for output
            X_axt2    = np.zeros((self.batch_size, self.img_x, self.img_y, self.img_z, 1), dtype="float32")
            X_diff    = np.zeros((self.batch_size, self.img_x, self.img_y, self.img_z, 2), dtype="float32")
            labels = np.zeros((self.batch_size), dtype='float32')

            for jj in range(0, self.batch_size):  
                X_axt2[jj], X_diff[jj], labels[jj]  =  self.read_hdf5_pirads(curr_vol_idx[jj])
 
            if not self.pretrainMode:
                # Binarize the labels at PIRADS cut off of three
                labels[labels <= self.pirads_cutoff] = 0
                labels[labels > self.pirads_cutoff] = 1

            if self.contrast_type == 'axt2':
                img = X_axt2
            elif self.contrast_type == 'diff':
                img = X_diff
            else:
                img = [X_axt2, X_diff]
            return img, labels
     
      
    def read_hdf5_pirads(self, vol_idx):
        filename = str(self.pirads_list[vol_idx]) + '.h5'
        return self.get_images(filename)
     
    def min_max_norm(self,img):
        ''' min max normalization'''
        max_ = img.max()
        min_ = img.min()
        img = (img-min_)/(max_-min_)
        img = img.astype('float32')
        return img
    
    def zmean_norm(self,img):
        ''' zmean normalization'''
        mean_ = img.mean()
        std_ = img.std()
        img = (img-mean_)/std_
        img = img.astype('float32')
        return img
    
    def get_images(self, filename):
        ''' Filename corresponds to an accession number '''
        if self.normalization == 'minmax':
            norm_fn = self.min_max_norm
        else:
            norm_fn = self.zmean_norm
    
        with h5py.File(self.data_dir + filename, 'r') as f:
            t2 = f['axt2'][:]
            adc = f['adc'][:]
            b1500 = f['b1500'][:]
            max_pirads = int(f.attrs['maxPIRADS'])
        
        # Process t2w data
        t2 = norm_fn(t2)
        t2_crop = self.crop_bbox(t2)
        if self.is_train:
            t2 = self.aug.augmentDatav2(t2_crop.squeeze())
        else:
            t2 =  t2_crop.squeeze()
        axt2_img = np.expand_dims(t2, axis=-1)
        
        # Process diffusion data
        adc = norm_fn(adc)
        b1500 = norm_fn(b1500)
        adc_crop = self.crop_bbox(adc)
        b1500_crop = self.crop_bbox(b1500)
        diff = np.stack((adc_crop,b1500_crop), axis=-1)
        if self.is_train:
            diff_img = self.aug.augmentDatav4(diff)
        else:
            diff_img = diff
        return axt2_img, diff_img, max_pirads
    
    def crop_bbox(self,img):
        ''' Use 3D crop function to retain the center of 3D volume.
        Resampling to 256x256x24 and retaining the center 16 slices
        '''
        cropped_img = resize(img, (256,256,self.bbox_shape[2]+8))
        cropped_img = myCrop3D(cropped_img,(self.bbox_shape[0],self.bbox_shape[1]))
        cropped_img = cropped_img[...,4:-4]
        return cropped_img
 



    


  

 
   
 
    
     
  
    
  
    