
'''
July8,2024 changes: Changing temporal embedding to correspond to age
Aug 25 changes: Updated representation learner weights
'''

MAX_TIME = 10    # Max age in years - Follow up years
MAX_CONTRAST = 2  # axT2 and Diffusion
MAX_FOLLOWUP = 5  # Excluding Baseline years
aux_dim = 5

FOLLOWUP_YR = 2
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-6 
CLASS_WTS = [0.5,2.0]
EPOCHS = 200
INITIAL_EPOCH = 0

# DATA DIRECTORIES
# location of h5 files
data_dir =  '/location/of/data/directory/containing/h5/files'
# location of training csv file
csv_dir = '/location/of/csv/directory/'
# location where results need to be saved
save_dir = '/location/where/results/need/to/be/saved/'

 
# IMAGE INFORMATION
img_size_x   = 128
img_size_y   = 128
img_size_z   = 16
batch_size = 32
num_features = 512

axt2_x = 128
axt2_y = 128
axt2_z = 16
axt2_channels = 1

diff_x = 128
diff_y = 128
diff_z = 16
diff_channels = 2


no_filters    = [1, 16, 32, 64, 64]
num_decs      = len(no_filters) - 1
pow           = 2**(num_decs-1)
latent_shape  = (img_size_x//pow,img_size_y//pow,img_size_z//pow,no_filters[-1])  # shape of latent volume for decoder
act_name      = 'relu'
bbox_shape    = (img_size_x,img_size_y,img_size_z) 
drop_PH       = True
num_timepoints = 1
num_contrasts = 2   # t2w and diffusion
num_sequence = num_timepoints * num_contrasts

 
 
# TRANSFORMER TRANING
max_time = MAX_TIME
max_contrast = MAX_CONTRAST
max_followup = MAX_FOLLOWUP
num_classes = 2       # 0: no PCa, 1: PCa  (PCa: PIRADS >= 4)
 

# TRANSFORMER ENCODER
pretr_latent_dim = 256
num_hidden_layers = 1
num_heads = 4
hidden_dim = 128
embedding_dim = hidden_dim
layer_norm_eps = 1e-6
attn_dropout_prob = 0.1


dropout_prob = 0.2
dropout_fc = 0.1
weight_decay = 1e-4
beta_1 = 0.9
beta_2 = 0.999
