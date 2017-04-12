# -*- coding: utf-8 -*-
import re
import MeCab
import glob

def file2list(filename):
    lookup = ('utf_8', 'euc_jp', 'euc_jis_2004', 'euc_jisx0213',
            'shift_jis', 'shift_jis_2004','shift_jisx0213',
            'iso2022jp', 'iso2022_jp_1', 'iso2022_jp_2', 'iso2022_jp_3',
            'iso2022_jp_ext','latin_1', 'ascii')
    for encode in lookup:
        with open(filename, mode='rb') as f:
            try:
                line = f.read()
                line = line.decode(encode)
                line = line.encode('utf-8')
                line = line.decode('utf-8')
                lines = line.split('\r\n')
                return lines
            except Exception as e:
                continue


def cleansing_data(lines):
    def filtering(string):
        if string == '':
            return False
        elif string[0] == '＠':
            return False
        elif string == '＜笑い＞':
            return False
        else:
            return True


    def cleansing(string):
        string = string.replace('＜笑い＞', '')
        string = re.sub('[\u3000]', '', string)
        if '：' in string:
            return string.split('：')[1]
        else:
            return string

    return list(map(cleansing, filter(filtering, lines)))


def parsed2data(string):
    def extract_word(line):
        return line.split('\t')[0]

    def filter_word(line):
        if line == 'EOS':
            return False
        else:
            return True

    lines = string.split('\n')
    return ' '.join(list(map(extract_word, filter(filter_word, lines))))



def main():
    with open('./data/train/master_data.txt', 'w') as f:
        files = glob.glob('./data/raw/nuc/*.txt')
        for _file in files:
            lines = file2list(_file)
            lines = cleansing_data(lines)
            m = MeCab.Tagger('-Ochasen')
            for line in lines:
                parsed = m.parse(line)
                line = parsed2data(parsed)
                f.write(line)
                f.write('\n')

    train_in = open('./data/train/train_in.txt', 'w')
    train_out = open('./data/train/train_out.txt', 'w')
    test_in = open('./data/train/test_in.txt', 'w')
    test_out = open('./data/train/test_out.txt', 'w')

    test_files = {'in': test_in, 'out': test_out}
    train_files = {'in': train_in, 'out': train_out}

    line_count = len(open('./data/train/master_data.txt').readlines())

    for i, line in enumerate(open('./data/train/master_data.txt')):
        files = train_files if i < (line_count / 2) else test_files
        if i % 2 == 0:
            files['in'].write(line)
        else:
            files['out'].write(line)
    train_in.close()
    train_out.close()
    test_in.close()
    test_out.close()


if __name__ == '__main__':
    main()