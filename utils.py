import tensorflow as tf
import numpy as np

def normalize_tensor_by_index(tensor,datatype):
    if datatype == "COLORS" or datatype == "COLORS_SHAPES":
        num_tensors = tensor.shape[-1]
        intervals = np.linspace(0, 1, num_tensors+1)
        normalized_tensors = []
        for i in range(num_tensors):
            t = tensor[..., i]
            lower = intervals[i]
            upper = intervals[i+1]
            t_norm = (t - tf.reduce_min(t)) / (tf.reduce_max(t) - tf.reduce_min(t))
            t_norm = t_norm * (upper - lower) + lower
            normalized_tensors.append(t_norm)
        return tf.stack(normalized_tensors, axis=-1)
    else:
       return tensor