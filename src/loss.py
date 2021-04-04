import tensorflow as tf

#Avoiding infitinte gradient for 0 in square root.
EPS = 1e-7

def N_tuple_loss(nb_patches, alpha=1, theta=0.35):
    @tf.function
    def loss(M, pred):
        diff = pred[:,:,tf.newaxis] - pred[:,tf.newaxis] # (N, nb_patches, nb_patches, features)
        D = tf.reduce_sum(diff*diff, axis=-1) # (N, nb_patches, nb_patches)
        D = tf.sqrt(D + EPS)
        norm_M = tf.reduce_sum(M, axis=[1,2])# (N,)
        loss_1 = (M*D) / norm_M[:, tf.newaxis, tf.newaxis]
        loss_2 = tf.nn.relu(theta - (1-M)*D) / (nb_patches*nb_patches-norm_M)[:,tf.newaxis,tf.newaxis]
        return loss_1 + alpha*loss_2
    return loss
    
def N_tuple_loss_pair(nb_patches, alpha=1, theta=0.35):
    @tf.function
    def loss(M, pred):
        diff = pred[:,0,:,tf.newaxis] - pred[:,1,tf.newaxis] # (N, nb_patches, nb_patches, features)
        D = tf.reduce_sum(diff*diff, axis=-1) # (N, nb_patches, nb_patches)
        D = tf.sqrt(D + EPS)
        norm_M = tf.reduce_sum(M, axis=[1,2]) + 1# (N,), avoiding overflow with + 1
        loss_1 = (M*D) / norm_M[:, tf.newaxis, tf.newaxis]
        loss_2 = tf.nn.relu(theta - (1-M)*D) / (nb_patches*nb_patches-norm_M)[:,tf.newaxis,tf.newaxis]
        return loss_1 + alpha*loss_2
    return loss
    