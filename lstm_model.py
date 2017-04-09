# -*- coding: utf-8 -*-

import tensorflow as tf

class NWModel(object):
    def __init__(self
    , train_input
    , batch_size
    , num_steps
    , hidden_size
    , vocab_size
    , num_layers
    ):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.size = hidden_size
        self.vocab_size = vocab_size
        self.cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True) for _ in range(num_layers)], state_is_tuple=True)
        self.state = self.cell.zero_state(batch_size, tf.float32)
        self.input_data, self.targets = self.ptb_producer([1 for _ in range(100000)], batch_size, num_steps)

    def logits(self):
        outputs = []
        embedding = tf.get_variable("embedding", [self.vocab_size, self.size], tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        softmax_w = tf.get_variable("softmax_w", [self.size, self.vocab_size], tf.float32)
        softmax_b = tf.get_variable("softmax_b", [self.vocab_size], tf.float32)
        for time_step in range(self.num_steps):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            (cell_output, state) = self.cell(inputs[:, time_step, :], self.state)
            outputs.append(cell_output)
        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, self.size])
        return tf.matmul(output, softmax_w) + softmax_b


    def loss(self, logits, targets, weights):
        loss = tf.contrib.seq2seq.sequence_loss(logits, targets, weights)
        return tf.reduce_sum(loss) / batch_size

#self.learning_rateを定義すること
    def training(self, cost):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        return optimizer

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    def ptb_producer(self, raw_data, batch_size, num_steps, name=None):
        with tf.name_scope("PTBProducer"):
            raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
            data_len = tf.size(raw_data)
            batch_len = data_len // batch_size
            data = tf.reshape(raw_data[0 : batch_size * batch_len],[batch_size, batch_len])
            epoch_size = (batch_len - 1) // num_steps
            i = 1
            x = tf.strided_slice(data, [0, i * num_steps],[batch_size, (i + 1) * num_steps])
            x.set_shape([batch_size, num_steps])
            y = tf.strided_slice(data, [0, i * num_steps + 1],[batch_size, (i + 1) * num_steps + 1])
            y.set_shape([batch_size, num_steps])
            return x, y
