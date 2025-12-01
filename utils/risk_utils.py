'''
Loss functions for the risk assessment module
Author: LU
'''
import tensorflow as tf
import numpy as np


class custom_BCE(tf.keras.losses.Loss):
    ''' Calculates custom BCE loss for a specific followup year
    Both y_true and y_pred have followup information i.e. both are Nx3 dim'''
    def __init__(self, 
                 followup_yr=0,
                ):
        super(custom_BCE, self).__init__()
        self.followup_yr = followup_yr
        self.bce = tf.keras.losses.BinaryCrossentropy()

    # Compute loss
    def call(self, y_true, y_pred):
        followup_yr = self.followup_yr
        # Extract a specific follow-up year for loss calculation
        y_pred = tf.gather(y_pred, np.arange(followup_yr,followup_yr+1), axis=1)
        loss = self.bce(y_true, tf.squeeze(y_pred))
        return loss

class WeightedCategoricalCrossEntropy(tf.keras.losses.Loss):
    ''' Calculates weighted categorical cross entropy loss for a specific followup year
    Both y_true and y_pred have followup information'''
    def __init__(self, 
                 followup_yr=0,
                 class_wts=None,
                 ):
        super(WeightedCategoricalCrossEntropy, self).__init__()
        self.followup_yr = followup_yr
        # self.cce = tf.keras.losses.CategoricalCrossentropy()
        self.class_wts = class_wts
 
    # Compute loss
    def call(self, y_true, y_pred):
        followup_yr = self.followup_yr
        y_pred = tf.gather(y_pred, np.arange(followup_yr,followup_yr+1), axis=1)
        y_true = tf.concat([1-y_true,y_true], axis=-1)
        y_pred = tf.concat([1-y_pred,y_pred], axis=-1)
        # Compute cross entropy
        loss_term = y_true * tf.math.log(y_pred) * self.class_wts
        loss_term = tf.reduce_sum(loss_term,axis=-1)
        loss = -tf.reduce_mean(loss_term)
        # loss = self.cce(y_true, y_pred)
        return loss


class MaskedWeightedCategoricalCrossEntropy(tf.keras.losses.Loss):
    ''' Masked weighted cross entropy loss for all years upto the specified followup_yr
    followup_yr = [0,1,2] for current risk, risk in 2-years, and risk in 5-years
    class_wts = [0.5, 2.0] weights for no PCa vs PCa classes
    year_wts = [1., 1., 1.] weights for each followup_yr

    y_pred - predictions from the model (batch size x 3) The three channels correspond to current risk, risk within 2 years, risk within 5 years
    y_true - target (batch size x 3 x 2) 
    channel 0 y_true : diagnosis labels (batch size x 3)
    channel 1 y_true: associated mask labels indicating if the assessment had an associated MRI exam (batch size x 3)

    '''
    def __init__(self, followup_yrs=3, class_wts=None, year_wts=[1.,1.,1.]):
        super(MaskedWeightedCategoricalCrossEntropy, self).__init__()
        self.class_wts = class_wts 
        self.epsilon = tf.constant(tf.keras.backend.epsilon())
        self.followup_yrs = followup_yrs
        self.year_wts = year_wts
 
    # Compute loss
    def call(self, target, y_pred):
        y_true, y_mask = tf.split(target, axis=-1,num_or_size_splits=2)
        y_mask = tf.cast(tf.squeeze(y_mask), tf.float32)    
        y_true = tf.cast(tf.squeeze(y_true), tf.float32)  
        loss = 0.0
        for idx in range(self.followup_yrs+1):
            curr_y_pred = tf.gather(y_pred, np.arange(idx,idx+1), axis=1)
            curr_y_true = tf.gather(y_true, np.arange(idx,idx+1), axis=1)
            curr_y_mask = tf.squeeze(tf.gather(y_mask,np.arange(idx,idx+1), axis=1))
            idx2retain = tf.stop_gradient(tf.squeeze(tf.where(curr_y_mask == 1)))
            if not tf.equal(tf.size(idx2retain), 0):
                curr_y_true = tf.gather(curr_y_true, idx2retain)
                curr_y_pred = tf.gather(curr_y_pred, idx2retain)
                loss_per_yr = self.single_timept_loss(curr_y_true, curr_y_pred)
            else:
                loss_per_yr = 0.0
            loss = loss + (loss_per_yr * self.year_wts[idx])
        loss = loss / self.followup_yrs
        return loss
    
    def single_timept_loss(self, y_true, y_pred):
        y_true = tf.concat([1-y_true,y_true], axis=-1)
        y_pred = tf.concat([1-y_pred,y_pred], axis=-1)
        # Compute cross entropy
        loss_term = y_true * -tf.math.log(y_pred) * self.class_wts
        loss_term = tf.reduce_sum(loss_term, axis=-1)
        loss = tf.reduce_mean(loss_term)
        return loss
    
class custom_AUC(tf.keras.metrics.Metric):
    '''
    Calculates custom AUC metrics
    '''
    def __init__(self, 
                 followup_yr=0,
                 name='AUC_eval', **kwargs):
        super(custom_AUC, self).__init__(**kwargs)
        self.followup_yr = followup_yr
        self.auc = tf.keras.metrics.AUC()
        self.auc_vals = self.add_weight(initializer='zeros', name='auc')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
      followup_yr = self.followup_yr
      y_pred = tf.gather(y_pred, np.arange(followup_yr,followup_yr+1), axis=1)
      auc_val = self.auc(y_true, tf.squeeze(y_pred))
      self.auc_vals.assign(auc_val)

    def reset_state(self):
       self.auc_vals.assign(0)
 
    def result(self):
      return self.auc_vals
    
    
def myCrop3D(ipImg,opShape):
    xDim,yDim = opShape
    zDim = ipImg.shape[2]
    opImg = np.zeros((xDim,yDim,zDim))
    
    xPad = xDim - ipImg.shape[0]
    yPad = yDim - ipImg.shape[1]
    
    x_lwr = int(np.floor(np.abs(xPad)/2))
    x_upr = int(np.ceil(np.abs(xPad)/2))
    y_lwr = int(np.floor(np.abs(yPad)/2))
    y_upr = int(np.ceil(np.abs(yPad)/2))
    if xPad >= 0 and yPad >= 0:
        opImg[x_lwr:xDim - x_upr ,y_lwr:yDim - y_upr,:] = ipImg
    elif xPad < 0 and yPad < 0:
        xPad = np.abs(xPad)
        yPad = np.abs(yPad)
        opImg = ipImg[x_lwr: -x_upr ,y_lwr:- y_upr,:]
    elif xPad < 0 and yPad >= 0:
        xPad = np.abs(xPad)
        temp_opImg = ipImg[x_lwr: -x_upr,:,:]
        opImg[:,y_lwr:yDim - y_upr,:] = temp_opImg
    else:
        yPad = np.abs(yPad)
        temp_opImg = ipImg[:,y_lwr: -y_upr,:]
        opImg[x_lwr:xDim - x_upr,:,:] = temp_opImg
    return opImg

def restoreCrop3D(ipMask,targetShape):
    xDim,yDim,zDim = ipMask.shape
    opImg = np.zeros((targetShape))
    
    xPad = xDim - targetShape[0]
    yPad = yDim - targetShape[1]
    
    x_lwr = int(np.floor(np.abs(xPad)/2))
    x_upr = int(np.ceil(np.abs(xPad)/2))
    y_lwr = int(np.floor(np.abs(yPad)/2))
    y_upr = int(np.ceil(np.abs(yPad)/2))
    
    if xPad >= 0 and yPad >= 0:
        opImg = ipMask[x_lwr:xDim - x_upr ,y_lwr:yDim - y_upr,:]
    elif xPad < 0 and yPad < 0:
        xPad = np.abs(xPad)
        yPad = np.abs(yPad)
        opImg[x_lwr: -x_upr ,y_lwr:- y_upr,:] = ipMask
    elif xPad < 0 and yPad >= 0:
        xPad = np.abs(xPad)
        
        temp_opImg= ipMask[:,y_lwr:yDim - y_upr,:] 
        opImg[x_lwr: -x_upr,:,:] = temp_opImg
    else:
        yPad = np.abs(yPad)
        temp_opImg = ipMask[x_lwr:xDim - x_upr,:,:]
        opImg[:,y_lwr: -y_upr,:] = temp_opImg
    return opImg

def contrastStretching(img, mask, lwr_prctile, upr_prctile):
    from skimage.exposure import rescale_intensity as rescale
    mm = img[mask > 0]
    p10 = np.percentile(mm,lwr_prctile)
    print('lwr_prctile',p10)
    p100 = np.percentile(mm,upr_prctile)
    print('upr_prctile',p100)
    opImg = rescale(img,in_range=(p10,p100))  
    return opImg


 

def generate_label_per_bin(temporal_labels,mask_labels,bin_indices=[2,5]):
    ''' General PCA labels and mask labels for temporal bins. 
    for e.g., [0] for current risk,[1-3] for risk in 2 years and [3-6] risk in 5 years
    The csv files have dx per year. This script combines them into three bins
    temporal_labels - labels for each year y_t depending on whether an MRI
                    exam corresponding to that year was positive (1) or negative (0)
    mask_labels - mask labels for each year y_t depending on whether an MRI exam was available that year.
                    For a negative exam, mask is 0 when no negative MRI available. For a positive exam,
                    mask is set to 1 even if no positive MRI available that year, as long as there was a prior positive MRI
    bin_indices - temporal bins. [2,5] are indices for risk in 2 years and risk in 5 years in addition to current risk
    
    '''
    batch_size = temporal_labels.shape[0]
    new_temporal_labels = np.zeros((batch_size,len(bin_indices)+1)) 
    new_mask_labels = np.zeros((batch_size,len(bin_indices)+1)) 
    bin_init_idx = 0
    # Add labels and masks for the current risk
    new_temporal_labels[:,bin_init_idx] = temporal_labels[:,bin_init_idx]
    new_mask_labels[:,bin_init_idx] = mask_labels[:,bin_init_idx]
    
    for batch_idx in range(batch_size):
        bin_init_idx = 1
        for bin_idx in range(len(bin_indices)):
            new_temporal_labels[batch_idx,bin_idx+1] = np.any(temporal_labels[batch_idx,bin_init_idx:bin_indices[bin_idx]+1])
            new_mask_labels[batch_idx,bin_idx+1] = np.any(mask_labels[batch_idx,bin_init_idx:bin_indices[bin_idx]+1])
            bin_init_idx = bin_indices[bin_idx]+1

    return new_temporal_labels, new_mask_labels