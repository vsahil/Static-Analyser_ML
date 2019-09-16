import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq import basic_rnn_seq2seq as seq_seq
from tensorflow.contrib.rnn import LSTMBlockCell as lstm
import time, sys
sys.path.append("../")

start = time.time()
encoder_inputs = []
decoder_inputs = []
for i in range(350):  
    encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                              name="encoder{0}".format(i)))

for i in range(45):
    decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                         name="decoder{0}".format(i)))
end1 = time.time()
print("Checkpoint model", end1-start)
start_time = time.time()
import check
model = seq_seq(encoder_inputs, decoder_inputs, lstm(512))