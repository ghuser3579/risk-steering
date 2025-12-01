'''
Layer definitions for the different models
Author: LU

'''

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

    
def encoder_block_contract(block_input, num_fts, pool_flag=True, block_name='1', act_name='relu'):        
    ''' CNN encoder block definition '''
    conv_kernel = (3,3,3)
    kernel_init = tf.keras.initializers.HeNormal(seed=1)
    num_groups = num_fts // 4
    if pool_flag:
        block_input = tf.keras.layers.MaxPooling3D(pool_size=(2,2,2), name='enc_mp_' + block_name)(block_input)
        
    down = tf.keras.layers.Conv3D(num_fts, conv_kernel, padding='same', name = 'enc_conv1_'+ block_name, kernel_initializer=kernel_init)(block_input)
    down = tfa.layers.GroupNormalization(groups = num_groups, name = 'enc_gn1_'+ block_name)(down)
    down = tf.keras.layers.Activation(act_name,name = 'enc_act1_'+ block_name)(down)
    down = tf.keras.layers.Conv3D(num_fts, conv_kernel, padding='same', name = 'enc_conv2_'+ block_name, kernel_initializer=kernel_init)(down)
    down = tfa.layers.GroupNormalization(groups = num_groups, name = 'enc_gn2_'+ block_name)(down)
    down = tf.keras.layers.Activation(act_name, name = 'enc_act2_'+ block_name)(down)
    return down

def set_encoder(args, ip_shape=None, act_name=None, name='enc'):
    ''' CNN Encoder definition'''
    no_filters =  args.no_filters
    num_levels = len(no_filters)
    if ip_shape == None:
        inputs = tf.keras.layers.Input((args.img_size_x, args.img_size_y, args.img_size_z, args.num_channels))
    else:
        inputs = tf.keras.layers.Input(ip_shape)
    if act_name == None:
        act_name = args.act_name  

    enc_op = encoder_block_contract(inputs, no_filters[1], pool_flag=False, block_name= name + '_' +  str(1), act_name=act_name)
    for level_idx in range(2,num_levels):
        enc_op = tf.keras.layers.Dropout(args.dropout_prob)(enc_op)
        enc_op = encoder_block_contract(enc_op, no_filters[level_idx], block_name=name + '_' + str(level_idx), act_name=act_name)
    model  = tf.keras.Model(inputs=[inputs], outputs=[enc_op], name=name + '_enc')
    return model

def set_projection_head(args, name='PH'):
    ''' Non-linear projection head'''
    num_decs = len(args.no_filters) - 1
    pow = 2**(num_decs-1)
    name=name
    ip_shape = (args.img_size_x//pow)*(args.img_size_y//pow)*(args.img_size_z//pow)*(args.no_filters[-1]) 
    inputs = tf.keras.layers.Input(ip_shape)
    PH_a = tf.keras.layers.Dense(1024, name = name +'_PH_a', activation='relu', use_bias=False)(inputs)
    PH_a = tf.keras.layers.Dropout(args.dropout_prob)(PH_a)
    PH_b = tf.keras.layers.Dense(args.num_features, name = name + '_PH_b', activation=None, use_bias=False)(PH_a) 
    PH = tf.keras.Model(inputs=[inputs], outputs=[PH_b], name=name + '_PH')
    return PH

def initialize_contrast_learner(args, data_type='axt2'):
    '''
    Contrast learner definition. UNET encoder followed by a projection head to transform
    specific MR contrast to a latent representation as specified by the parameters in args
    Inputs: MR image (Num volumes x Height x Width x Depth x Channels)
    Channels: 1 for T2WI, 2 for DWI
    '''
    if data_type == 'axt2':
      args.num_channels = 1
      args.input_shape = (args.img_size_x,args.img_size_y,args.img_size_z,1)
      args.contrast_type = 'axt2'
    else:
      args.num_channels = 2
      args.input_shape = (args.img_size_x,args.img_size_y,args.img_size_z,2)
      args.contrast_type = 'diff'
    print(f'Instantiating {data_type} encoder')
    enc_model = set_encoder(args, name=data_type)
    projection_head  = set_projection_head(args, name=data_type)
    input_x = enc_model.inputs   
    arr_x  = enc_model(input_x)
    g_x    = tf.keras.layers.Flatten()(arr_x)
    g_x = tf.keras.layers.Dropout(args.dropout_fc)(g_x)
    head_x = projection_head(g_x)
    CL_model = tf.keras.Model(input_x, head_x, name='contrast_learner_' + data_type)
    print('Model inputs:', CL_model.inputs)
    print('Model outputs:', CL_model.outputs)
    return CL_model


def set_RL_projection_head(args, name = ''):
  ''' Non-linear projection head'''
  inputs = tf.keras.layers.Input((args.hidden_dim))
  PH_a = tf.keras.layers.Dense(args.hidden_dim, name = name + '_PH_fc1', activation='relu', use_bias=False)(inputs)
  PH_a = tf.keras.layers.Dropout(0.2)(PH_a)
  PH_b = tf.keras.layers.Dense(args.hidden_dim, name = name + '_PH_fc2', activation=None, use_bias=False)(PH_a) 
  PH = tf.keras.Model(inputs=[inputs], outputs=[PH_b], name= name + '_PH_fc')
  return PH

def initialize_representation_learner(args):
    """
    Representation Learner definition.
    Model takes T2WI (Nx128x128x16x1) and DWI (Nx128x128x16x2) with two-channel ADC+B1500 images as inputs 
    and generates a latent representation Nx512
    """
    # contrast specific learners
    axt2_learner = initialize_contrast_learner(args, data_type='axt2')
    diff_learner = initialize_contrast_learner(args, data_type='diff')
    input_axt2 = axt2_learner.inputs
    input_diff = diff_learner.inputs
    hidden_axt2 = axt2_learner(input_axt2)
    hidden_diff = diff_learner(input_diff)
    # concatenate representations along axis=1 (Nx2x512)
    hidden = CustomConcatenate(args, axis=1, name='RL')(hidden_axt2,hidden_diff)
    # Add position embedding T2WI or DWI
    hidden = TransformerPositionEmbedder(args, scale=False)(hidden)
    hidden = EncoderTransformers(args, name='RL')(hidden)
    # Attention pooling to transform (Nx2x256) to (Nx256)
    hidden = SimpleAttentionPool(args)(hidden)
    # Non linear projection head
    hidden_PH  = set_RL_projection_head(args, name='RL')(hidden)
    imageRL = tf.keras.Model([input_axt2,input_diff], hidden_PH, name='imageRL')
    print(f'Instantiated Representation Learner')
    return imageRL

class Cumulative_Probability_Layer(tf.keras.layers.Layer):
  ''' Layer to estimate cumulative probability of predicting outcome
  The layers are set such that 
  layer 0: baseline risk
  layer 1: marginal increase in risk Year 2
  layer 2: marginal increase in risk Year 5  

  Adapted from OncoNet - MIRAI for Mammo
  '''
  def __init__(self, max_followup, 
               layer_type='risk_pred',
               kernel_init=tf.keras.initializers.HeNormal(seed=1)):
    super(Cumulative_Probability_Layer, self).__init__()
    self.max_followup = max_followup
    self.layer_type = layer_type
    mask = tf.ones([self.max_followup, self.max_followup])
    mask = tf.linalg.band_part(mask, -1, 0)
    self.upper_triagular_mask = tf.Variable(tf.transpose(mask), trainable=False, name='upper_triangular_mask')
    self.base_hazard_fc = tf.keras.layers.Dense(1, kernel_initializer=kernel_init, name='baseline_prob')   # baseline hazard
    self.hazard_fc = [tf.keras.layers.Dense(1, kernel_initializer=kernel_init, name=f'marginal_prob_{i}') for i in range(self.max_followup)]  # marginal increases in hazard  
     
    self.relu = tf.keras.layers.Activation('relu')
    if layer_type == 'risk_pred':
        self._name = "CumulProbLayer" + 'RP'
    else:
       self._name = "CumulProbLayer" + 'RF'
       
  def hazards(self, x):
    pos_hazard = []
    for _, layer_module in enumerate(self.hazard_fc ):
      raw_hazard = layer_module(x)
      raw_hazard = self.relu(raw_hazard)
      pos_hazard.append(raw_hazard)
    pos_hazard = tf.concat(pos_hazard, axis=-1)
    return pos_hazard

  def call(self, x):
    base_hazard =  self.base_hazard_fc(x) 
    hazards = self.hazards(x)
    hazards = tf.tile(hazards[...,None],[1,1,self.max_followup])  #expanded_hazards is (B,T, T)
    masked_hazards = hazards * self.upper_triagular_mask # masked_hazards now (B,T,T)
    cumul_hazard = tf.reduce_sum(masked_hazards,axis=1) + base_hazard
    total_hazard = tf.concat([base_hazard,cumul_hazard],axis=1)
    return tf.ensure_shape(total_hazard,[None, self.max_followup+1])

def initialize_risk_prediction_model(args, max_followup, longitudinal_data_flag=True, activation=False):
    ''' Risk prediction model variant to instantiate both risk prediction and risk refinement modules 
        based on longitudinal_data_flag value. Activation controls sigmoid activation to convert hazards to probabilities
    '''
    if longitudinal_data_flag:
      input_z = tf.keras.Input(shape=(args.aux_dim))
      cumul_layer = Cumulative_Probability_Layer(max_followup=max_followup, layer_type='risk_refine')
      model_name = 'risk_refine_model'
    else:
      input_z = tf.keras.Input(shape=(args.hidden_dim))
      cumul_layer = Cumulative_Probability_Layer(max_followup=max_followup, layer_type='risk_pred')
      model_name = 'risk_pred_model'
    cumul_hazard = cumul_layer(input_z)
    if activation:
      cumul_hazard = tf.keras.layers.Activation('sigmoid')(cumul_hazard)
    return tf.keras.Model(input_z, cumul_hazard, name=model_name)

def initialize_temporal_learner(args, num_classes=5):
    '''
    Temporal learner model
    '''
    input_z_prev = tf.keras.Input(shape=args.pretr_latent_dim)   # previous representation
    input_time_prev = tf.keras.Input(shape=(1), dtype=tf.int32)   # time of previous representation

    input_z_curr = tf.keras.Input(shape=args.pretr_latent_dim)   # current representation
    input_time_curr = tf.keras.Input(shape=(1), dtype=tf.int32)  # time of current representation
    
    prev_input = [input_z_prev,input_time_prev]
    curr_input = [input_z_curr,input_time_curr]

    hidden = CustomConcatenate(args, axis=1, name='TL')(input_z_prev,input_z_curr)
    time_info = tf.keras.layers.Concatenate(axis=-1)([input_time_prev,input_time_curr])
    hidden = Temporal_Embedder(args, scale=False, name='temporal_embed_i')(hidden,time_info)  # position encoding
    hidden = EncoderTransformers(args, name='TL')(hidden)
    hidden = tf.keras.layers.GlobalAveragePooling1D (data_format='channels_last',name='GAP')(hidden)
    hidden = tf.keras.layers.Dense(num_classes, name='temporal_PH')(hidden)           # change signal
    print('Initializing temporal representation learner')
    temporal_ai = tf.keras.Model([prev_input,curr_input], [hidden], name='temporal_ai')
    print(f'Model inputs: {temporal_ai.inputs}')
    print(f'Model outputs: {temporal_ai.outputs}')
    return temporal_ai

class CustomConcatenate(tf.keras.layers.Layer):
        ''' Custom concatenation layer'''
        def __init__(self, args, axis=1, name='1'):
            super(CustomConcatenate, self).__init__()
            self.args = args
            self.axis=axis
            self.concat = tf.keras.layers.Concatenate(axis=self.axis)
            self._name = 'CustomConcatenate' + name
        
        def call(self,x, y):
            if self.axis==1:
                x = tf.expand_dims(x,axis=1)   #(N,1,D)
                y = tf.expand_dims(y,axis=1)   #(N,1,D)
            latent = self.concat([x,y])    #(N,2,D)
            return latent
        
class Temporal_Embedder(tf.keras.layers.Layer):
    ''' Transformer embedder with sin cos positional embedding for time'''
    def __init__(self, args, scale=False, **kwargs):
    
        super().__init__(**kwargs)
        self.args = args
        self.scale = scale
        self.projection = tf.keras.layers.Dense(args.embedding_dim) 
        self.embed_add_fc = tf.keras.layers.Dense(args.embedding_dim)
        self.embed_scale_fc = tf.keras.layers.Dense(args.embedding_dim)
    
    def build(self, input_shape):
        self.batch_size = self.args.batch_size
        self.time_embed = self.add_weight(
            shape=(1, self.args.MAX_TIME, self.args.embedding_dim),
            initializer="zeros",
            trainable=False,  # fixed sin-cos embedding
            name="temporal_embeddings",
        )
        positions = tf.cast(tf.range(start=0, limit=self.args.MAX_TIME, delta=1),tf.float32)
        time_embed = get_1d_sincos_pos_embed_from_grid(self.args.embedding_dim, positions, add_cls_token=False)[None,...]
        self.time_embed.assign(time_embed)

        super().build(input_shape)
     
    def condition_on_pos_embed(self, x, embed):
        if self.scale:
            embedded_x = self.embed_scale_fc(embed) * x + self.embed_add_fc(embed)
            return embedded_x
        else:
            return  x + embed
        
    def call(self, x, time_seq):
        batch_size = shape_list(x)[0]
        time_embed = tf.tile(self.time_embed,[batch_size,1,1])
        time_embed = tf.gather(time_embed, indices=time_seq, axis=1, batch_dims=1)
        x = self.condition_on_pos_embed(self.projection(x), time_embed)
        return x
    
class EncoderTransformers(tf.keras.layers.Layer):
    ''' Transformer layers with multi head self-attention '''
    def __init__(self, args, name = '', **kwargs):
        super().__init__(**kwargs)
        self.args = args
        self.layer = [TransformerLayer(args, name=name + f"_tx_enc_layer_{i}") for i in range(args.num_hidden_layers)]
        self.layer_norm_fc = tf.keras.layers.LayerNormalization(epsilon=self.args.layer_norm_eps, name= name + '_tx_enc_norm')
        self._name =  "TransformerEncoder" + '_' + name
    def call(self, x):
        ''' Computes a forward pass through the model'''
        for _, layer_module in enumerate(self.layer):
            x = layer_module(x)
        x = self.layer_norm_fc(x)
        return x
    
class TransformerLayer(tf.keras.layers.Layer):
    ''' Basic transformer layer with multi-head attention'''
    def __init__(self, args, name='T_Encoder_layer', **kwargs):

        super().__init__(**kwargs)

        self.args = args
        self.multihead_attention = tf.keras.layers.MultiHeadAttention(num_heads=self.args.num_heads, 
                                                       key_dim=self.args.hidden_dim, 
                                                       dropout=self.args.attn_dropout_prob)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=self.args.layer_norm_eps)
        self.fc1 = tf.keras.layers.Dense(2 * self.args.hidden_dim, activation = tf.nn.gelu)
        self.fc2 = tf.keras.layers.Dense(self.args.hidden_dim, activation = tf.nn.gelu)
        self.layer_norm_fc = tf.keras.layers.LayerNormalization(epsilon=self.args.layer_norm_eps)
        self.dropout = tf.keras.layers.Dropout(self.args.attn_dropout_prob) 
        self._name = name

    def multilayer_perceptron(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def call(self, x):
        # Layer normalization 1
        x1 = self.layer_norm(x)
        attention_output = self.multihead_attention(x1,x1)
        # skip connection
        x2 = tf.keras.layers.Add()([attention_output, x])
        # Layer normalization 2
        x3 = self.layer_norm(x2)
        x3 = self.multilayer_perceptron(x3)
        x = tf.keras.layers.Add()([x3, x2])
        return x
    
def shape_list(tensor):
    """
    Deal with dynamic shape in tensorflow cleanly.
    """
    if isinstance(tensor, np.ndarray):
        return list(tensor.shape)

    dynamic = tf.shape(tensor)
    if tensor.shape == tf.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, add_cls_token=False):
    """
    From: https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit_mae/modeling_tf_vit_mae.py
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = tf.range(embed_dim // 2, dtype="float32")
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = tf.reshape(pos, [-1])  # (M,)
    out = tf.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    # half of the positions get sinusoidal pattern and the rest gets
    # cosine pattern and then they are concatenated
    emb_sin = tf.sin(out)  # (M, D/2)
    emb_cos = tf.cos(out)  # (M, D/2)

    emb = tf.concat([emb_sin, emb_cos], axis=1)  # (M, D)
    if add_cls_token:
        pos_embed = tf.concat([tf.zeros((1, embed_dim)), pos_embed], axis=0)
    return emb

class TransformerPositionEmbedder(tf.keras.layers.Layer):
    ''' Transformer layer to add contrast embeddings to an input'''
    def __init__(self, args, scale=False, **kwargs):
        
        super().__init__(**kwargs)
        self.args = args
        self.scale = scale
        self.projection = tf.keras.layers.Dense(args.embedding_dim)   # adding a projection layer
        self.embed_add_fc = tf.keras.layers.Dense(args.embedding_dim)
        self.embed_scale_fc = tf.keras.layers.Dense(args.embedding_dim)
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self._name = "PositionEmbedder"

    def build(self, input_shape):
        self.batch_size = self.args.batch_size
        self.pos_embed = self.add_weight(
            shape=(1, self.args.max_contrast, self.args.embedding_dim),
            initializer="zeros",
            trainable=False,  # fixed sin-cos embedding
            name="position_embeddings",
        )
        positions = tf.cast(tf.range(start=0, limit=2, delta=1),tf.float32)
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.args.embedding_dim, positions, add_cls_token=False)[None,...]
        self.pos_embed.assign(pos_embed)

        super().build(input_shape)
    
    def condition_on_pos_embed(self, x, embed):
        if self.scale:
            embedded_x = self.embed_scale_fc(embed) * x + self.embed_add_fc(embed)
            return embedded_x
        else:
            return  x + embed
    
    def call(self, x):
        ''' Computes a forward pass through the model'''
        batch_size = shape_list(x)[0]
        embed = tf.tile(self.pos_embed,[batch_size,1,1])   # (N,2,D) embedding for contrast
        x = self.condition_on_pos_embed(self.projection(x), embed)   # N, 2, D
        return x
    
class SimpleAttentionPool(tf.keras.layers.Layer):
    ''' Attention pooling along the channel dimension: similar to OncoNet
    Also refer to: https://github.com/SHI-Labs/Compact-Transformers/blob/main/src/utils/transformers.py'''
    def __init__(self, args):
        super(SimpleAttentionPool, self).__init__()

        self.args = args
        self.attention_fc = tf.keras.layers.Dense(1)
        self.softmax = tf.keras.layers.Softmax(axis=1)
        self._name = "SimpleAttnPoolv2"
    
    def call(self,x):
        attention_scores = self.softmax(self.attention_fc(x))     
        x = tf.matmul(tf.transpose(attention_scores,[0,2,1]), x)       
        x = tf.squeeze(x, axis=1)                           
        return x
