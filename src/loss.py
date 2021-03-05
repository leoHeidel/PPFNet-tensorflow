import tensorflow as tf

def N_tuple_loss(nb_patches, alpha=1, theta=0.35):
    @tf.function
    def loss(M, pred):
        diff = pred[:,:,tf.newaxis] - pred[:,tf.newaxis] # (N, nb_patches, nb_patches, features)
        D = tf.norm(diff*diff, axis=-1) # (N, nb_patches, nb_patches)
        norm_M = tf.reduce_sum(M, axis=[1,2])# (N,)
        loss_1 = (M*D) / norm_M[:, tf.newaxis, tf.newaxis]
        loss_2 = tf.nn.relu(theta - (1-M)*D) / (nb_patches*nb_patches-norm_M)
        return loss_1 * alpha*loss_2
    return loss
    