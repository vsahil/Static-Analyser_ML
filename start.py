# import tensorflow as tf

# def var(initial_value=None,
#     trainable=None,
#     collections=None,
#     validate_shape=True,
#     caching_device=None,
#     name=None,
#     variable_def=None,
#     dtype=None,
#     expected_shape=None,
#     import_scope=None,
#     constraint=None,
#     use_resource=None,
#     synchronization=tf.VariableSynchronization.AUTO,
#     aggregation=tf.VariableAggregation.NONE,
#     shape=None):
    
#     if shape == None:
#         if initial_value == None:
#             print("Empty initial value")
#             assert False
#         else:
#             return initial_value
#     else:
#         return shape

# def rand_normal_(shape,
#     mean=0.0,
#     stddev=1.0,
#     dtype=tf.dtypes.float32,
#     seed=None,
#     name=None):
#     try:
#         assert(shape != None)
#     except AssertionError as e:
#         e.args += ('some other', 'important', 'information', 42)
#         raise
#     return shape

def random_randn(*argv):
    if len(argv) == 0:
        return (1,)        # shape of the output is (1,)
    else:
        return [i for i in argv]

# def matmul(a, b):       # a and b can be multidimensional matrices (a and b are just the shapes)
#     assert(a[-1] == b[-2])      # last element of the first matches the second last element of second
#     def checkEqual(L1, L2):
#         return len(L1) == len(L2) and sorted(L1) == sorted(L2)
#     assert checkEqual(a[:-2], b[:-2])       # handles multidimensional matrices
#     a[:-2].extend([a[-2], b[-1]])     # (In-place operation) leaving the last 2 dimensions remain same, and then last 2 dimension are multiplied as normal
#     print(a)
#     return a
