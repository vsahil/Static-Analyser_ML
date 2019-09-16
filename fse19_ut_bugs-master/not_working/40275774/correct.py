import tensorflow as tf
import tempfile
from IPython import embed

sequences = [[1, 2, 3], [4, 5, 1], [1, 2]]
label_sequences = [[0, 1, 0], [1, 0, 0], [1, 1]]

def make_example(sequence, labels):

    ex = tf.train.SequenceExample()

    sequence_length = len(sequence)
    ex.context.feature["length"].int64_list.value.append(sequence_length)

    fl_tokens = ex.feature_lists.feature_list["tokens"]
    fl_labels = ex.feature_lists.feature_list["labels"]
    for token, label in zip(sequence, labels):
        fl_tokens.feature.add().int64_list.value.append(token)
        fl_labels.feature.add().int64_list.value.append(label)
    return ex


writer = tf.python_io.TFRecordWriter('./test.tfrecords')
for sequence, label_sequence in zip(sequences, label_sequences):
    ex = make_example(sequence, label_sequence)
    writer.write(ex.SerializeToString())
writer.close()

tf.reset_default_graph()

file_name_queue = tf.train.string_input_producer(['./test.tfrecords'], num_epochs=None)

reader = tf.TFRecordReader()
##reader = tf.TFRecordReader.read(queue, name=None)


context_features = {
    "length": tf.FixedLenFeature([], dtype=tf.int64)
}
sequence_features = {
    "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
}

ex = reader.read(file_name_queue)

# Parse the example (returns a dictionary of tensors)
context_parsed, sequence_parsed = tf.parse_single_sequence_example(
    serialized=ex.value,
    context_features=context_features,
    sequence_features=sequence_features
)


context = tf.contrib.learn.run_n(context_parsed, n=1, feed_dict=None)
print(context[0])
sequence = tf.contrib.learn.run_n(sequence_parsed, n=1, feed_dict=None)
print(sequence[0])
