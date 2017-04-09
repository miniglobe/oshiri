# -*- coding: utf-8 -*-
"""
- init_scale - 重みの初期尺度
- learning_rate - 学習率の初期値
- max_grad_norm - 勾配の最大許容ノルム
- num_layers - LSTM層の数
- num_steps - LSTMのアンロールされたステップの数
- hidden_size - LSTMユニットの数
- max_epoch - 最初の学習率で訓練されたエポックの数
- max_max_epoch - トレーニングのエポックの総数
- keep_prob - ドロップアウト層にウェイトを保持する確率
- lr_decay - "max_epoch"後の各エポックの学習率の減衰
- batch_size - バッチサイズ
"""
import tensorflow as tf
import input_reader
import lstm_model

def main():
    with tf.Graph().as_default():
        train_data = input_reader.raw_data("data/ptb.train.txt")

        m = lstm_model.NWModel(
            train_data
            , batch_size = 64
            , num_steps = 20
            , hidden_size = 200
            , vocab_size = 10000
            , num_layers = 2
            )

        logits = m.logits()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(m.input_data))
            print(sess.run(logits))


if __name__ == '__main__':
    main()
