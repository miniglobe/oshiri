# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile

from tensorflow.python.platform import gfile
import tensorflow as tf
import MeCab

# Special vocabulary symbols - we always put them at the start.
_MYSELF = b"_MYSELF"
_OPPONENT = b"_OPPONENT"
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

MYSELF_ID = 0
OPPONENT_ID = 1
PAD_ID = 2
GO_ID = 3
EOS_ID = 4
UNK_ID = 5

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):

  print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
  vocab = {}
  with gfile.GFile(data_path, mode="rb") as f:
    counter = 0
    for line in f:
      counter += 1
      if counter % 1000 == 0:
        print("  processing line %d" % counter)
      line = tf.compat.as_bytes(line)
      tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
      for w in tokens:
        word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
        if word in vocab:
          vocab[word] += 1
        else:
          vocab[word] = 1
    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    if len(vocab_list) > max_vocabulary_size:
      vocab_list = vocab_list[:max_vocabulary_size]
    with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
      for w in vocab_list:
        vocab_file.write(w + b"\n")

def morpheme_line(line, m):

  res = m.parse(line)
  l = res.split('\n')
  one_line = ""
  for e in l:
      one_line += e.split('\t')[0] + ' '
  one_line = one_line.replace('EOS','')
  return one_line

def morpheme_vocablary(data_path, vocablary_path=None):

  m = MeCab.Tagger()
  with tf.gfile.GFile(data_path, mode="r") as data_file:
      line = data_file.readline()
      vocab_list = ""
      while line:
        one_line = morpheme_line(line, m)
        vocab_list += one_line.strip() + '\n'
        line = data_file.readline()
      with gfile.GFile(vocablary_path, mode="wb") as vocab_file:
          vocab_file.write(vocab_list)

def initialize_vocabulary(vocabulary_path):

  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):

  print("Tokenizing data in %s" % data_path)
  vocab, _ = initialize_vocabulary(vocabulary_path)
  with gfile.GFile(data_path, mode="rb") as data_file:
    with gfile.GFile(target_path, mode="w") as tokens_file:
      counter = 0
      for line in data_file:
        counter += 1
        if counter % 1000 == 0:
          print("  tokenizing line %d" % counter)
        token_ids = sentence_to_token_ids(tf.compat.as_bytes(line), vocab,
                                          tokenizer, normalize_digits)
        tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

def prepare_data(data_dir, vocabulary_size, tokenizer=None):

  # Create vocabularies of the appropriate sizes.
  train_data_path = os.path.join(data_dir, "train_data.txt")
  test_data_path = os.path.join(data_dir, "test_data.txt")
  train_morpheme_path = os.path.join(data_dir, "vocab_morpheme.train")
  test_morpheme_path = os.path.join(data_dir, "vocab_morpheme.test")
  train_vocab_path = os.path.join(data_dir, "vocab%d.train" % vocabulary_size)
  test_vocab_path = os.path.join(data_dir, "vocab%d.test" % vocabulary_size)

  morpheme_vocablary(train_data_path, train_morpheme_path)
  morpheme_vocablary(test_data_path, test_morpheme_path)

  create_vocabulary(train_vocab_path, train_morpheme_path , vocabulary_size, tokenizer)
  create_vocabulary(test_vocab_path, test_morpheme_path , vocabulary_size, tokenizer)

  # Create token ids for the training data.
  train_ids_path = train_data_path + ".ids"
  data_to_token_ids(train_morpheme_path, train_ids_path, train_vocab_path, tokenizer)

  # Create token ids for the development data.
  dev_ids_path = test_data_path + ".ids"
  data_to_token_ids(test_morpheme_path, dev_ids_path, test_vocab_path, tokenizer)

  return (train_ids_path, dev_ids_path, train_vocab_path, test_vocab_path)
