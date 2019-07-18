import tensorflow as tf

def var(initial_value=None,
    trainable=None,
    collections=None,
    validate_shape=True,
    caching_device=None,
    name=None,
    variable_def=None,
    dtype=None,
    expected_shape=None,
    import_scope=None,
    constraint=None,
    use_resource=None,
    synchronization=tf.VariableSynchronization.AUTO,
    aggregation=tf.VariableAggregation.NONE,
    shape=None):
    
    if shape == None:
        if initial_value == None:
            assert False
        else:
            return initial_value
    else:
        return shape

def rand_normal_(shape,
    mean=0.0,
    stddev=1.0,
    dtype=tf.dtypes.float32,
    seed=None,
    name=None):
    
    assert(shape) != None
    return shape
