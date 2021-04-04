import tensorflow as tf
import tensorflow.keras as keras

class PPFNet:
    def make_model(self):
        inputs = keras.Input(shape=self.input_shape) # (batch, patches, points_per_patches, ppf_features)
        x = inputs 
        
        #Patch pointNet
        for units in self.point_net_units[:-1]:
            x = keras.layers.Dense(units)(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation(keras.activations.relu)(x)
        x = keras.layers.Dense(self.point_net_units[-1])(x)
        x = tf.reduce_max(x, axis=-2)# (batch, patches, patche_features)
        local_features = x
        
        #Computing global features, nothing in the loop for orinigal implementation
        for i,units in enumerate(self.global_units):
            x = keras.layers.Dense(units)(x)
            if i != len(self.global_units) -1:
                if self.use_batch_norm:
                    x = keras.layers.BatchNormalization()(x)
                x = keras.layers.Activation(keras.activations.relu)(x)            
        x = tf.reduce_max(x, axis=-2)# (batch, patches)
        
        #Combining features
        x = tf.repeat(x[...,tf.newaxis,:], tf.shape(local_features)[-2], axis=-2)
        x = tf.concat([local_features, x], axis=-1)
        
        #Computing combined features
        for units in self.mlp_units[:-1]: 
            x = keras.layers.Dense(units)(x)
            if self.use_batch_norm:
                x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation(keras.activations.relu)(x)
        x = keras.layers.Dense(self.point_net_units[-1])(x)
        
        return keras.models.Model(inputs, x)
    
    def __init__(self, 
                 point_net_units=[32,32,32], 
                 mlp_units=[64,64], 
                 global_units=[],
                 use_batch_norm=True,
                 input_shape = (None, None,10)
                ):
        self.point_net_units = point_net_units
        self.mlp_units = mlp_units
        self.global_units = global_units
        self.use_batch_norm = use_batch_norm
        self.input_shape = input_shape
        self.model = self.make_model()