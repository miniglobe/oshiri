# -*- coding: utf-8 -*-

import tensorflow as tf

class Input(object):
	def __init__(self
	, data
	, barch_size = 64
	, num_steps = 20
	):
		self.batch_size = batch_size = config.batch_size
		self.num_steps = num_steps = config.num_steps
		self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
		self.input_data, self.targets = NWModel.ptb_producer(data, batch_size, num_steps, name=name)

class NWModel(object):
	def __init__(self
	, batch_size
	, num_steps
	, hidden_size
	, vocab_size
	, num_layers
	, input_
	, keep_prob
	, is_training
	):
		self._input = input_
		self.batch_size = input_.batch_size
		self.num_steps = input_.num_steps
		self.size = hidden_size
		self.vocab_size = vocab_size
		self.is_training = is_training #訓練用かどうか

		cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True) for _ in range(config.num_layers)], state_is_tuple=True)
		self._initial_state = cell.zero_state(batch_size, data_type())

	def loss(self, logits):
		pass

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
