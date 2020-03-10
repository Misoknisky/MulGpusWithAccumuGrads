#coding=utf-8
'''
Created on 2020年3月4日
@author: Administrator
@email: 1113471782@qq.com
'''
import tensorflow as tf
def _deduplicate_indexed_slices(values, indices):
    """Sums `values` associated with any non-unique `indices`.
    Args:
      values: A `Tensor` with rank >= 1.
      indices: A one-dimensional integer `Tensor`, indexing into the first
      dimension of `values` (as in an IndexedSlices object).
    Returns:
      A tuple of (`summed_values`, `unique_indices`) where `unique_indices` is a
      de-duplicated version of `indices` and `summed_values` contains the sum of
      `values` slices associated with each unique index.
    """
    unique_indices, new_index_positions = tf.unique(indices)
    summed_values = tf.unsorted_segment_sum(
      values, new_index_positions,
      tf.shape(unique_indices)[0])
    return (summed_values, unique_indices)
def create_train_op(optim_type,learning_rate):
    if optim_type == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif optim_type == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
    elif optim_type == 'rprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
    elif optim_type == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise NotImplementedError('Unsupported optimizer: {}'.format(optim_type))
    return optimizer
def average_gradients(tower_grads):
    # calculate average gradient for each shared variable across all GPUs
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grad_name, grad_value = grad_and_vars[0]
        if grad_name is None:
            # no gradient for this variable, skip it
            average_grads.append((grad_name, grad_value))
            continue

        if isinstance(grad_name, tf.IndexedSlices):
            # If the gradient is type IndexedSlices then this is a sparse
            #   gradient with attributes indices and values.
            # To average, need to concat them individually then create
            #  a new IndexedSlices object.
            indices = []
            values = []
            for g, v in grad_and_vars:
                indices.append(g.indices)
                values.append(g.values)
            all_indices = tf.concat(indices, 0)
            avg_values = tf.concat(values, 0) / len(grad_and_vars)
            # deduplicate across indices
            av, ai = _deduplicate_indexed_slices(avg_values, all_indices)
            grad = tf.IndexedSlices(av, ai, dense_shape=grad_name.dense_shape)

        else:
            # a normal tensor can just do a simple average
            grads = []
            for g, v in grad_and_vars:
                if g is not None:
                # Add 0 dimension to the gradients to represent the tower.
                   expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over 
                   grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

        # the Variables are redundant because they are shared
        # across towers. So.. just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)

        average_grads.append(grad_and_var)

    assert len(average_grads) == len(list(zip(*tower_grads)))
    
    return average_grads
        
