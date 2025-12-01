'''
This module contain all the augmentations that are used in Data generator.
Script supports augmentations for single channel and multi-channel inputs
'''
 
import scipy  
import numpy as np
import tensorflow as tf
import elasticdeform
import tensorflow_addons as tfa
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
 

class AugmentObject:
    def __init__(self, args):
        self.args = args
        self._set_transformations()

    def _set_transformations(self):
        # single channel augmentations
        self.available_transformations = { 'fliplr': self.random_flipLR,
                                        'crop'    : self.crop_resize,
                                        'cont_stretch': self.contrast_stretch,
                                        'rotate'  : self.random_rotate,
                                        'no_action': self.no_action
        }
        # multi-channel augmentations (ADC+b1500)
        self.mc_available_transformations = { 'fliplr': self.random_flipLR_mc,
                                            'cont_stretch': self.contrast_stretch_mc,
                                            'rotate'  : self.random_rotate,
                                            'no_action': self.no_action
        }
        return  


    def _get_transformation(self):
        ''' Select a random transformation to apply for an image'''
        tx_to_apply = np.random.choice(list(self.available_transformations), 1)   
        return tx_to_apply
    
    def _get_mc_transformation(self):
        ''' Select a random transformation to apply for an image'''
        tx_to_apply = np.random.choice(list(self.mc_available_transformations), 1)   
        return tx_to_apply

    def augmentData(self, x, y):
        tx_to_apply = self._get_transformation()
        x_aug = self.available_transformations[tx_to_apply[0]](x)
        tx_to_apply = self._get_transformation()
        y_aug = self.available_transformations[tx_to_apply[0]](x)
        return x_aug, y_aug
    
    def augmentDatav1(self, x):
        ''' Apply a random transformation'''
        tx_to_apply = self._get_transformation()
        x = self.available_transformations[tx_to_apply[0]](x)
        return x
    
    def augmentDatav2(self, x):
        ''' Apply a random transformation before elastic deformation'''    
        tx_to_apply = self._get_transformation()
        x = self.available_transformations[tx_to_apply[0]](x)
        x = self.random_elasticdeformation(x.numpy())
        return tf.identity(x)
    
    def augmentDatav3(self, x):
        ''' Apply two random transformations for the same input. Useful for simple contrastive learning'''
        tx_to_apply = self._get_transformation()
        x_aug = self.available_transformations[tx_to_apply[0]](x)
        tx_to_apply = self._get_transformation()
        y_aug = self.available_transformations[tx_to_apply[0]](x)
        y_aug = self.random_elasticdeformation(y_aug.numpy())
        return x_aug, tf.identity(y_aug)
    
    def augmentDatav4(self, x):
        ''' Apply a random transformation for multi-channel inputs'''    
        tx_to_apply = self._get_mc_transformation()
        x = self.mc_available_transformations[tx_to_apply[0]](x)
        return  x


    def crop_resize(self, x):
        ''' Perform central crop'''
        min_val = x.min()
        max_val = x.max()
        x_tmp = tf.identity(x)
        x_tmp = tf.image.central_crop(x_tmp, self.args.central_fraction)
        x_tmp = tf.image.resize(x_tmp,(self.args.img_size_x, self.args.img_size_y), method='bilinear')
        return tf.clip_by_value(x_tmp, min_val, max_val) 

    def contrast_stretch(self,x, lwr_prctile=10,upr_prctile=99.99):
        ''' Histogram-based contrast stretching.'''
        mm = np.ndarray.flatten(x)
        p10 = np.percentile(mm,lwr_prctile)
        p100 = np.percentile(mm,upr_prctile)
        x_aug = np.clip(x,p10,p100)
        return tf.identity(x_aug)   

    def contrast_stretch_mc(self,x, lwr_prctile=10,upr_prctile=99.99):
        ''' Histogram-based multi-channel contrast stretching'''
        x_aug = np.zeros(x.shape)
        for idx in range(x.shape[-1]):
            x_aug[...,idx] = self.contrast_stretch(x[...,idx],lwr_prctile,upr_prctile)
        return tf.identity(x_aug)   

    def no_action(self,x):
        ''' Perform no augmentation'''
        return tf.identity(x)

    def random_rotate(self,x):
        ''' in-plane rotation randomly sampled from angles'''
        angles = [-15,15]
        random_angle = np.random.uniform(angles[0], angles[1])
        min_val = x.min()
        img = scipy.ndimage.rotate(x, reshape=False, 
                                   angle=random_angle, axes=(1, 0),
                                   order=1, mode='reflect',
                                   cval=min_val)
        return tf.identity(img)
    
    def random_flipLR(self, x):
        ''' Random left right flip'''
        return tf.image.random_flip_left_right(x)
    
    def random_elasticdeformation(self,x):
        ''' Generate random in-plane elastic deformations'''
        min_val = x.min()
        max_val = x.max()
        sigma = self.args.deform_sigma
        axis = self.args.deform_axis
        e_img = elasticdeform.deform_random_grid(x, sigma=sigma, points=5, order=3, mode='reflect',cval = x.min(), axis=axis)
        return tf.clip_by_value(e_img, min_val, max_val)
    
    def random_flipLR_mc(self, x):
        x_aug = np.zeros(x.shape)
        for idx in range(x.shape[-1]):
            x_aug[...,idx] = np.fliplr(x[...,idx])
        return tf.identity(x_aug)
     
    def random_flipUD(self, x):
        return tf.image.random_flip_up_down(x)
        
    

    
 





