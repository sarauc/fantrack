import tensorflow as tf
    
def pool_layer(x):
    return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def unravel_argmax(argmax, shape):
        output_list = []
        output_list.append(argmax // (shape[2] * shape[3]))
        output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
        return tf.stack(output_list)
    
    
def unpool_layer2x2(x, raveled_argmax, out_shape):
        argmax = unravel_argmax(raveled_argmax, tf.to_int64(out_shape))
        output = tf.zeros([out_shape[1], out_shape[2], out_shape[3]])

        height = tf.shape(output)[0]
        width = tf.shape(output)[1]
        channels = tf.shape(output)[2]

        t1 = tf.to_int64(tf.range(channels))
        t1 = tf.tile(t1, [((width + 1) // 2) * ((height + 1) // 2)])
        t1 = tf.reshape(t1, [-1, channels])
        t1 = tf.transpose(t1, perm=[1, 0])
        t1 = tf.reshape(t1, [channels, (height + 1) // 2, (width + 1) // 2, 1])

        t2 = tf.squeeze(argmax)
        t2 = tf.stack((t2[0], t2[1]), axis=0)
        t2 = tf.transpose(t2, perm=[3, 1, 2, 0])

        t = tf.concat([t2, t1], 3)
        indices = tf.reshape(t, [((height + 1) // 2) * ((width + 1) // 2) * channels, 3])

        x1 = tf.squeeze(x)
        x1 = tf.reshape(x1, [-1, channels])
        x1 = tf.transpose(x1, perm=[1, 0])
        values = tf.reshape(x1, [-1])

        delta = tf.SparseTensor(indices, values, tf.to_int64(tf.shape(output)))
        return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_reorder(delta)), 0)


def max_indices(maps, height, width):
    ''' 
        find the maximum predictions in output maps of each targets.
        :Arguments:
              output_map: the output maps from the assocnet or the  labels.
              height: the height of the maps.
              width: the width of the maps.
        :Return:
              row and column indices of maximum entries in each output map.
    '''
    
    with tf.name_scope('max_indices'):
        
        num_pixels = height*width
        
        # flatten the maps/labels for each target
        flat_maps = tf.reshape(maps,[num_pixels, -1],'flat_maps')

        # find maximums for each targets        
        indices = tf.argmax(flat_maps, axis = 0)
            
        # find the row and column indices of the maximums 
        row_idx = tf.floordiv(indices, width)
        
        col_idx =  tf.floormod(indices, height)

        row_idx = tf.cast(row_idx, tf.int32)
        col_idx = tf.cast(col_idx, tf.int32)

        return row_idx, col_idx
        

def correct_predictions(x_pred, y_pred, x_true, y_true):
    '''
        find the correct predictions (or associations) done for each targets
        :Arguments:
            x_pred,y_pred: the row and column indices of the maximums 
            in each output map.
            x_true,y_true: the row and column indices of the true 
            associations.
        :Return: A Tensor consisting of 1s and 0s for True and False
                 predictions, respectively.
    '''

    with tf.name_scope('correct_predictions'):
        
        tf_pred = tf.logical_and(tf.equal(x_pred, x_true), 
                                 tf.equal(y_pred, y_true), 
                                 'true_false_predictions'
                                )

        return tf.cast(tf_pred, tf.float32)
        