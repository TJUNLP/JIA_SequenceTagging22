
import codecs, re
import numpy as np

def get_word_index(files):

    source_vob = {}

    for testf in files:
        f = open(testf, 'r')
        fr = f.readlines()
        for line in fr:
            if line.__len__() <= 1:
                continue

            sourc = line.strip('\r\n').rstrip('\n').rstrip('\r').split(' ')
            lower_word = sourc[0].lower()
            if not bool(re.search('[a-z]', lower_word)):
                if bool(re.search('[0-9]', lower_word)):
                    continue
            if lower_word not in source_vob.keys():
                source_vob[lower_word] = 0
                # print(lower_word)

        f.close()

    return source_vob


def read2vec(file1, fastext_w2v_file, source_vob):
    num = 0
    print(file1)
    fr = open(file1, 'r')
    line = fr.readline()
    line = fr.readline()
    while line:

        if num >= len(source_vob):
            break

        line = line.strip('\r\n').rstrip('\n').rstrip('\r').rstrip(' ')
        # print('---' + line + '---')
        values = line.split()
        if len(values) > 301:
            line = fr.readline()
            continue

        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        # print(word)
        # print(coefs)

        if word in source_vob.keys():
            source_vob[word] = 1
            num += 1
            print(word)
            print('----------', len(source_vob), num)

        line = fr.readline()

    print(num)
    num0 = 0
    for i in source_vob.keys():
        if source_vob[i] == 0:
            if bool(re.search('[a-z]', i)):
                num0 += 1
            else:
                print(i)
                pass

    print(num0)


if __name__ == '__main__':

    # file1 = "./data/w2v/wiki.en.vec"
    file1 = "./data/w2v/glove.6B.100d.txt"
    fastext_w2v_file = "./data/w2v/fasttext.300d.txt"

    trainfile = "./data/CoNLL2003_NER/eng.train.BIOES.txt"
    devfile = "./data/CoNLL2003_NER/eng.testa.BIOES.txt"
    testfile = "./data/CoNLL2003_NER/eng.testb.BIOES.txt"
    files = [trainfile, devfile, testfile]
    source_vob = get_word_index(files)
    read2vec(file1, fastext_w2v_file, source_vob)
