# coding: utf-8
import sys
import MeCab

m = MeCab.Tagger ("-Ochasen")

print ("���̖��O�̓{�u�ł��B")
print m.parse("���̖��O�̓{�u�ł��B")