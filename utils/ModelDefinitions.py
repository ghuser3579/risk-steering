import tensorflow as tf
from utils.LayerDefinitions import initialize_representation_learner
from utils.LayerDefinitions import initialize_temporal_learner
from utils.LayerDefinitions import initialize_risk_prediction_model
 
class Context_aware_imaging_only_model(tf.keras.Model):
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
    
    def evaluate_risk(self, data):
        prior_img, prior_time, curr_img, curr_time = data 
        # get current MRI representation
        curr_zi = self.representation_learner(curr_img)
        # get current image risk
        initial_hazard = self.risk_prediction_model(curr_zi)
        initial_risk = tf.keras.layers.Activation('sigmoid')(initial_hazard)

        # get prior MRI representations
        prior_zi = self.representation_learner(prior_img)

        # create a temporal sequence of representations and times
        z_all = tf.expand_dims(tf.concat([prior_zi, curr_zi], axis=0),axis=0)
        t_all = tf.expand_dims(tf.concat([prior_time, curr_time], axis=0), axis=0)
        t_all = tf.cast(t_all, tf.int32)

        hidden = self.temporal_learner.get_layer('temporal_embed_i')(z_all, t_all)
        hidden = self.temporal_learner.get_layer('TransformerEncoder_TL')(hidden)
        hidden = self.temporal_learner.get_layer('GAP')(hidden)
        zi_agg = self.temporal_learner.get_layer('temporal_PH')(hidden)
        zi_agg = self.BN(zi_agg)
        i_change = self.risk_refinement_model(zi_agg)
        refined_hazard = tf.keras.layers.Add()([initial_hazard, i_change])
        refined_risk = tf.keras.layers.Activation('sigmoid')(refined_hazard)
        return initial_risk.numpy().squeeze(), refined_risk.numpy().squeeze()
        
 