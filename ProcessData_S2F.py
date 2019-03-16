#coding=utf-8
__author__ = 'JIA'
import numpy as np
import pickle
import json, math
import Seq2fragment



def get_data(trainfile,devfile, testfile,w2v_file, c2v_file, datafile, w2v_k=300, c2v_k=25, maxlen = 50):

    # 数据处理的入口函数

    word_vob, word_idex_word, target_vob, target_idex_word, max_s = get_word_index([trainfile, devfile, testfile])
    print("source vocab size: " + str(len(word_vob)))
    print("target vocab size: " + str(len(target_vob)))
    print("target vocab size: " + str(target_vob))
    print("target vocab size: " + str(target_idex_word))

    word_W, word_k= load_vec_txt(w2v_file, word_vob, k=w2v_k)
    print("word2vec loaded!")
    print("all vocab size: " + str(len(word_vob)))
    print("source_W  size: " + str(len(word_W)))

    char_vob, char_idex_char, max_c = get_Character_index({trainfile, devfile, testfile})
    print("source char size: ", char_vob.__len__())
    print("max_c: ", max_c)
    print("source char: " + str(char_idex_char))

    character_W, character_k = load_vec_character(c2v_file, char_vob, c2v_k)
    print('character_W shape:', character_W.shape)

    max_context = 0
    max_fragment = 1
    train_fragment_list, max_context, max_fragment = Seq2fragment.Seq2frag(trainfile, word_vob, target_vob, target_idex_word, max_context, max_fragment)
    dev_fragment_list, max_context, max_fragment = Seq2fragment.Seq2frag(devfile, word_vob, target_vob, target_idex_word, max_context, max_fragment)
    test_fragment_list, max_context, max_fragment = Seq2fragment.Seq2frag(testfile, word_vob, target_vob, target_idex_word, max_context, max_fragment)
    print('max_context--', max_context, 'max_fragment--', max_fragment)
    print('len(test_fragment_list)---', len(test_fragment_list))

    Type_idex_word = {0: 'LOC', 1: 'ORG', 2: 'PER', 3: 'MISC'}
    Type_vob = {'LOC': 0, 'ORG': 1, 'PER': 2, 'MISC': 3}

    train = make_idx_word_index(train_fragment_list, max_context, max_fragment)
    dev = make_idx_word_index(dev_fragment_list, max_context, max_fragment)
    test = make_idx_word_index(test_fragment_list, max_context, max_fragment)
    print(len(train), len(dev), len(test))

    chartrain = make_idx_char_index(train_fragment_list, max_context, max_fragment, max_c, char_vob, word_idex_word)
    chardev = make_idx_char_index(dev_fragment_list, max_context, max_fragment, max_c, char_vob, word_idex_word)
    chartest = make_idx_char_index(test_fragment_list, max_context, max_fragment, max_c, char_vob, word_idex_word)
    print(len(chartrain), len(chardev), len(chartest))

    print ("dataset created!")
    out = open(datafile,'wb')
    pickle.dump([train, dev, test,
                 chartrain, chardev, chartest,
                 word_vob, word_idex_word,
                 target_vob, target_idex_word,
                 Type_vob, Type_idex_word,
                 char_vob, char_idex_char,
                 word_W, word_k,
                 character_W, character_k,
                 max_context, max_fragment, max_c], out, 0)
    out.close()



def get_word_index(files):

    source_vob = {}
    target_vob = {}
    sourc_idex_word = {}
    target_idex_word = {}
    count = 1
    tarcount = 1

    max_s = 0

    if not target_vob.__contains__("O"):
        target_vob["O"] = tarcount
        target_idex_word[tarcount] = "O"
        tarcount += 1

    num = 0
    for file in files:
        f = open(file, 'r')
        fr = f.readlines()
        for line in fr:
            if line.__len__() <= 1:
                if num > max_s:
                    max_s = num
                # print(max_s, '  ', num)
                num = 0
                continue

            num += 1
            sourc = line.strip('\r\n').rstrip('\n').rstrip('\r').split(' ')

            if not source_vob.__contains__(sourc[0]):
                source_vob[sourc[0]] = count
                sourc_idex_word[count] = sourc[0]
                count += 1
            if not target_vob.__contains__(sourc[4]):
                target_vob[sourc[4]] = tarcount
                target_idex_word[tarcount] = sourc[4]
                tarcount += 1

        f.close()

    if not source_vob.__contains__("**UNK**"):
        source_vob["**UNK**"] = count
        sourc_idex_word[count] = "**UNK**"
        count += 1

    return source_vob, sourc_idex_word, target_vob, target_idex_word, max_s



def make_idx_char_index(fraglist, max_context, max_fragment, max_c, char_vob, word_idex_word):

    char_fragment_all = []
    char_leftcontext_all = []
    char_rightcontext_all = []

    for line in fraglist:

        fragment = line[0]
        context_left = line[2]
        context_right = line[3]

        data_fragment = []
        for wordindex in fragment:

            word = word_idex_word[wordindex]
            data_c = []
            for chr in range(0, min(word.__len__(), max_c)):
                if not char_vob.__contains__(word[chr]):
                    data_c.append(char_vob["**UNK**"])
                else:
                    data_c.append(char_vob[word[chr]])

            data_c = data_c + [0] * max(max_c - word.__len__(), 0)

            data_fragment.append(data_c)

        data_fragment = data_fragment + [[0] * max_c] * max(0, max_fragment - len(fragment))
        char_fragment_all.append(data_fragment)

        data_leftcontext = []
        for wordindex in context_left:

            word = word_idex_word[wordindex]
            data_c = []
            for chr in range(0, min(word.__len__(), max_c)):
                if not char_vob.__contains__(word[chr]):
                    data_c.append(char_vob["**UNK**"])
                else:
                    data_c.append(char_vob[word[chr]])

            data_c = data_c + [0] * max(max_c - word.__len__(), 0)

            data_leftcontext.append(data_c)

        data_leftcontext = data_leftcontext + [[0] * max_c] * max(0, max_context - len(context_left))
        char_leftcontext_all.append(data_leftcontext)

        data_rightcontext = []
        for wordindex in context_right:

            word = word_idex_word[wordindex]
            data_c = []
            for chr in range(0, min(word.__len__(), max_c)):
                if not char_vob.__contains__(word[chr]):
                    data_c.append(char_vob["**UNK**"])
                else:
                    data_c.append(char_vob[word[chr]])

            data_c = data_c + [0] * max(max_c - word.__len__(), 0)

            data_rightcontext.append(data_c)

        data_rightcontext = data_rightcontext + [[0] * max_c] * max(0, max_context - len(context_right))
        char_rightcontext_all.append(data_rightcontext)

    return [char_fragment_all, char_leftcontext_all, char_rightcontext_all]



def make_idx_word_index(fraglist, max_context, max_fragment):

    data_fragment_all = []
    data_leftcontext_all = []
    data_rightcontext_all = []
    data_t_all = []

    index2word_Type = {0: 'LOC', 1: 'ORG', 2: 'PER', 3: 'MISC'}
    word2index_Type = {'LOC': 0, 'ORG': 1, 'PER': 2, 'MISC': 3}

    for line in fraglist:

        fragment = line[0]
        fragment_tag = line[1]
        context_left = line[2]
        context_right = line[3]

        data_fragment = fragment + [0] * max(0, max_fragment-len(fragment))
        data_t = np.zeros(4)
        data_t[word2index_Type[fragment_tag]] = 1

        data_leftcontext = [0] * max(0, max_context-len(context_left)) + context_left
        data_rightcontext = context_right + [0] * max(0, max_context-len(context_right))

        data_fragment_all.append(data_fragment)
        data_leftcontext_all.append(data_leftcontext)
        data_rightcontext_all.append(data_rightcontext)
        data_t_all.append(data_t)

    return [data_fragment_all, data_leftcontext_all, data_rightcontext_all, data_t_all]




def get_Character_index(files):

    source_vob = {}
    sourc_idex_word = {}
    max_c = 18
    count = 1

    for file in files:
        f = open(file, 'r')
        fr = f.readlines()
        for line in fr:
            if line.__len__() <= 1:
                continue

            sourc = line.strip('\r\n').rstrip('\n').rstrip('\r').split(' ')[0]

            for character in sourc:
                if not source_vob.__contains__(character):
                    source_vob[character] = count
                    sourc_idex_word[count] = character
                    count += 1

        f.close()
    if not source_vob.__contains__("**PAD**"):
        source_vob["**PAD**"] = 0
        sourc_idex_word[0] = "**PAD**"

    if not source_vob.__contains__("**UNK**"):
        source_vob["**UNK**"] = count
        sourc_idex_word[count] = "**UNK**"
        count += 1

    return source_vob, sourc_idex_word, max_c


def load_vec_txt(fname, vocab, k=300):
    f = open(fname)
    w2v={}
    W = np.zeros(shape=(vocab.__len__() + 1, k))
    unknowtoken = 0
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        w2v[word] = coefs
    f.close()
    w2v["**UNK**"] = np.random.uniform(-0.25, 0.25, k)
    for word in vocab:
        lower_word = word.lower()
        if not w2v.__contains__(lower_word):
            w2v[word] = w2v["**UNK**"]
            unknowtoken +=1
            W[vocab[word]] = w2v[word]
        else:
            W[vocab[word]] = w2v[lower_word]

    print('UnKnown tokens in w2v', unknowtoken)
    return W, k


def load_vec_character(c2vfile, vocab_c_inx, k=50):

    fi = open(c2vfile, 'r')
    c2v = {}
    for line in fi:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        c2v[word] = coefs
    fi.close()

    c2v["**UNK**"] = np.random.uniform(-1*math.sqrt(3/k), math.sqrt(3/k), k)

    W = np.zeros(shape=(vocab_c_inx.__len__()+1, k))

    for i in vocab_c_inx:
        if not c2v.__contains__(i):
            c2v[i] = c2v["**UNK**"]
            W[vocab_c_inx[i]] = c2v[i]
        else:
            W[vocab_c_inx[i]] = c2v[i]

    return W, k


if __name__ == '__main__':

    w2v_file = "./data/w2v/glove.6B.100d.txt"
    c2v_file = "./data/w2v/C0NLL2003.NER.c2v_2.txt"
    trainfile = "./data/CoNLL2003_NER/eng.train.BIOES.txt"
    devfile = "./data/CoNLL2003_NER/eng.testa.BIOES.txt"
    testfile = "./data/CoNLL2003_NER/eng.testb.BIOES.txt"
    resultdir = "./data/result/"

    word_vob, word_idex_word, target_vob, target_idex_word, max_s = get_word_index([trainfile, devfile, testfile])
    print("source vocab size: " + str(len(word_vob)))
    print("target vocab size: " + str(len(target_vob)))
    print("target vocab size: " + str(target_vob))
    print("target vocab size: " + str(target_idex_word))


    max_context = 0
    max_fragment = 1
    test_fragment_list, max_context, max_fragment = Seq2fragment.Seq2frag(testfile, word_vob, target_vob,
                                                                          target_idex_word, max_context, max_fragment)
    print('max_context--', max_context, 'max_fragment--', max_fragment)
    print('len(test_fragment_list)---', len(test_fragment_list))