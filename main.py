# -*- coding: utf-8 -*-

import tensorflow as tf
import input_reader
import lstm_model

def main():
	train_data = input_reader.raw_data("data/ptb.train.txt")

if __name__ == '__main__':
	main()
