"""
Config file for pretraining 
@author: LU
"""

# DATA DIRECTORIES
# location of h5 files
data_dir =  '/location/of/data/directory/containing/h5/files'
# location of training csv file
csv_dir = '/location/of/csv/directory/'
# location where results need to be saved
save_dir = '/location/where/results/need/to/be/saved/'
 

# IMAGE INFORMATION
contrast_type = 'axt2'    # options are axt2 for training T2WI CNN, diff for training diff CNN, biparametric for training RL
normalization = 'minmax'  # options are 'minmax' and 'zmean'
max_contrast = 2
img_size_x   = 128
img_size_y   = 128
img_size_z   = 16
batch_size    = 16
num_features  = 512

axt2_x = 128
axt2_y = 128
axt2_z = 16
axt2_channels = 1

diff_x = 128
diff_y = 128
diff_z = 16
diff_channels = 2

BBOX_SHAPE = (128,128,16)

# CNN Parameters
no_filters    = [1, 16, 32, 64, 64]   # Kernel filters per resolution level
num_decs      = len(no_filters) - 1
pow           = 2**(num_decs-1)
latent_shape  = (img_size_x//pow,img_size_y//pow,img_size_z//pow,no_filters[-1])  # shape of latent volume for decoder
act_name      = 'relu'
bbox_shape    = (img_size_x,img_size_y,img_size_z) 
drop_PH       = True
num_timepoints = 1
num_contrasts = 2   # t2w and diffusion
 
 
# TRANSFORMER encoder parameters
pretr_latent_dim = 512     # latent dimension from CNN encoders
num_hidden_layers = 1
num_heads = 4
hidden_dim = 256          # latent dimension from image RL
embedding_dim = hidden_dim
layer_norm_eps = 1e-6
attn_dropout_prob = 0.1


''' Training params'''
initial_epoch = 0
epochs        = 100
lr            = 1e-4
lr_mode      = 'constant'
''' Loss params''' 
loss_type    = 2     
temperature  = 0.07 
dropout_prob = 0.2
dropout_fc = 0.1
weight_decay = 1e-4
beta_1 = 0.9
beta_2 = 0.999

 
 

 

 
