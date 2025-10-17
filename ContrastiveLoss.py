'''
Contains functions for variants of contrastive loss used in this work

'''

import tensorflow as tf
from keras import backend as K
import numpy as np
 
 
class Constrained_Contrastive_Loss:
    def __init__(self, cfg):
        self.cfg = cfg
        self.temperature  = cfg.temperature   
        self.batch_size   = cfg.batch_size
        self.num_features = cfg.num_features

    def calc_euclidean_loss(self, x,y):
        ''' Invariance loss: Calculate euclidean loss between unnormalized vectors'''
        mse = tf.keras.losses.MeanSquaredError()
        euc_loss = mse(x, y)
        return tf.reduce_mean(euc_loss)
    
    def cosine_similarity(self, vector_a, vector_b, temperature):
        '''
        Calculating cosine similarity between two l2-normalized vectors
        '''        
        norm_vector_a = tf.nn.l2_normalize(vector_a,axis=-1)
        norm_vector_b = tf.nn.l2_normalize(vector_b,axis=-1)
        cosine_similarity_value = tf.linalg.matmul(norm_vector_a,norm_vector_b,transpose_b=True)/temperature     
        return cosine_similarity_value
    
    def get_softmax(self, sim_metric):
        '''
        Compute probability associated with a similarity metric
        ''' 
        prob = tf.math.exp(sim_metric)    
        return prob
    
    def calc_pirads_constrained_contrastive_loss(self, labels, fts):
        '''
        Implementation of PI-RADS guided contrastive learning
        labels: PI-RADS assessments (1,2,4,5)
        fts:    latent representations generated for bpMRI image
        '''
        bs=self.batch_size
        net_global_loss=0.0
        temperature = self.temperature
        # iterate over each image in the batch
        for idx in range(0,bs):
            # current image idx and its corresponding label
            pos_idx = np.arange(idx, idx+1, dtype=np.int32)
            z_1 = tf.gather(fts, pos_idx)
            curr_label = tf.squeeze(tf.gather(labels, pos_idx))

            # exclude current index 
            pos_exclude_idx = np.delete(np.arange(bs),pos_idx)
            new_ft_1 = tf.gather(fts,pos_exclude_idx)
            new_labels = tf.gather(labels,pos_exclude_idx)

            # identify all positive idx that are same as current label
            pos_label_idx =  tf.where(new_labels == curr_label)
            # identify all negative idx for the current label
            neg_label_idx =  tf.where(new_labels != curr_label)
            # gather the positive features from one augmented version
            z_pos = tf.gather(new_ft_1, pos_label_idx)
            z_neg = tf.gather(new_ft_1, neg_label_idx)

            if not tf.equal(tf.size(z_pos), 0):
                z_pos_sim = self.cosine_similarity(z_1, z_pos, temperature)  
                z_neg_sim = self.cosine_similarity(z_1, z_neg, temperature) 
                neg_prob = tf.reduce_sum(tf.math.exp(z_neg_sim))
                pos_prob = tf.reduce_sum(tf.math.exp(z_pos_sim))
                den_term = pos_prob + neg_prob
                relative_probability = tf.divide(pos_prob, den_term)
                curr_loss = -tf.math.log(relative_probability) 
                net_global_loss = net_global_loss +  (curr_loss/bs)
        reg_cost = tf.reduce_sum(net_global_loss)
        return reg_cost


    