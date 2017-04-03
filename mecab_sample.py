# coding: utf-8
import sys
import MeCab

m = MeCab.Tagger ("-Ochasen")

print ("私の名前はボブです。")
print m.parse("私の名前はボブです。")