import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq import basic_rnn_seq2seq as seq_seq
from tensorflow.contrib.rnn import LSTMBlockCell as lstm
import time

start = time.time()
input_sequence_length = 512
encoder_inputs = []
decoder_inputs = []
for i in range(350):  
    encoder_inputs.append(tf.placeholder(tf.float32, shape=[None, input_sequence_length],
                                              name="encoder{0}".format(i)))

for i in range(45):
    decoder_inputs.append(tf.placeholder(tf.float32, shape=[None, input_sequence_length],
                                         name="decoder{0}".format(i)))
end1 = time.time()
print("Checkpoint model", end1-start)
model = seq_seq(encoder_inputs, decoder_inputs, lstm(512))
end2 = time.time()
print("Checkpoint eval", end2-end1)