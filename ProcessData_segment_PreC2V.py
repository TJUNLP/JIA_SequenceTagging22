#coding=utf-8
__author__ = 'JIA'

import numpy as np
import pickle
import json
import re
import math

def load_vec_pkl(fname,vocab,k=300):
    """
    Loads 300x1 word vecs from word2vec
    """
    W = np.zeros(shape=(vocab.__len__() + 1, k))
    w2v = pickle.load(open(fname,'rb'))
    w2v["UNK"] = np.random.uniform(-0.25, 0.25, k)
    for word in vocab:
        if not w2v.__contains__(word):
            w2v[word] = w2v["UNK"]
        W[vocab[word]] = w2v[word]
    return w2v,k,W


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
    return w2v, W

def load_vec_txt_all(fname,vocab,k=300):
    f = open(fname)
    w2v={}
    vocab_w2v = {}

    i = 0
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')

        if len(coefs) == k:
            w2v[word] = coefs
            vocab_w2v[str(word)] = i
            i += 1

    f.close()

    w2v["UNK"] = np.random.uniform(-0.25, 0.25, k)
    vocab_w2v["UNK"] = i

    W = np.zeros(shape=[i + vocab.__len__() + 1, k])


    for word in vocab:
        if not w2v.__contains__(word):
            w2v[word] = w2v["UNK"]
        W[vocab[word]] = w2v[word]

    for word in vocab_w2v:
        if not vocab.__contains__(word):
            vocab[word] = vocab.__len__() + 1
            W[vocab[word]] = w2v[word]

    W = W[:vocab.__len__()+1]


    return w2v, k, W, vocab


def load_vec_pos(vocab_c_inx, k=30):

    W = np.zeros(shape=(vocab_c_inx.__len__()+1, k))

    for i in vocab_c_inx:
        W[vocab_c_inx[i]] = np.random.uniform(-1*math.sqrt(3/k), math.sqrt(3/k), k)

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

    return W


def make_idx_data_index(file, max_s, source_vob, target_vob):

    data_s_all=[]
    data_tBIOES_all = []

    f = open(file,'r')
    fr = f.readlines()

    count = 0

    data_tBIOES = []
    data_s = []
    for line in fr:

        if line.__len__() <= 1:

            data_s = data_s + [0] * max(0, max_s - count)
            data_tBIOES = data_tBIOES + [[1] + [0] * 5] * max(0, max_s - count)

            data_s_all.append(data_s)

            data_tBIOES_all.append(data_tBIOES)

            data_tBIOES = []
            data_s = []
            count = 0
            continue

        sent = line.strip('\r\n').rstrip('\n').split(' ')
        if not source_vob.__contains__(sent[0]):
            data_s.append(source_vob["**UNK**"])
        else:
            data_s.append(source_vob[sent[0]])


        targetvecBIOES = np.zeros(5 + 1)
        targetvecBIOES[target_vob[sent[4][0]]] = 1
        data_tBIOES.append(targetvecBIOES)

        count += 1

    f.close()
    return [data_s_all, data_tBIOES_all]


def make_idx_character_index(file, max_s, max_c, source_vob):

    data_s_all=[]
    count = 0
    f = open(file,'r')
    fr = f.readlines()

    data_w = []
    for line in fr:

        if line.__len__() <= 1:
            num = max_s - count
            # print('num ', num, 'max_s', max_s, 'count', count)

            for inum in range(0, num):
                data_tmp = []
                for i in range(0, max_c):
                    data_tmp.append(0)
                data_w.append(data_tmp)
            # print(data_s)
            # print(data_t)
            data_s_all.append(data_w)

            data_w = []
            count =0
            continue

        data_c = []
        word = line.strip('\r\n').rstrip('\n').split(' ')[0]

        for chr in range(0, min(word.__len__(), max_c)):
            if not source_vob.__contains__(word[chr]):
                data_c.append(source_vob["**UNK**"])
            else:
                data_c.append(source_vob[word[chr]])

        num = max_c - word.__len__()
        for i in range(0, max(num, 0)):
            data_c.append(0)
        count +=1
        data_w.append(data_c)

    f.close()
    return data_s_all


def make_idx_POS_index(file, max_s, source_vob, Poswidth=3):

    width = (Poswidth-1)//2

    count = 0
    data_s_all = []
    data_s = []

    f = open(file,'r')
    fr = f.readlines()
    sen_i = 0

    for i, line in enumerate(fr):

        if line.__len__() <= 1:
            num = max_s - count
            # print('num ', num, 'max_s', max_s, 'count', count)
            for inum in range(0, num):
                data_s.append([0] * Poswidth)
            # print(data_s)
            # print(data_t)
            data_s_all.append(data_s)
            data_s = []
            count = 0
            sen_i = 0
            continue

        data_w = []


        for k in range(width, 0, -1):
            if sen_i - k < 0:
                data_w.append(0)
                # print('>>>')
            else:
                sourc_pre = fr[i - k].strip('\r\n').rstrip('\n').split(' ')[1]
                data_w.append(source_vob[sourc_pre])
                # print(sourc_pre)

        sent = line.strip('\r\n').rstrip('\n').split(' ')[1]
        if not source_vob.__contains__(sent):
            data_w.append(source_vob["**UNK**"])
        else:
            data_w.append(source_vob[sent])
        # print(sent)

        for k in range(1, width+1):
            if i + k >= fr.__len__() or fr[i + k].__len__() <= 1:
                for s in range(k, width+1):
                    data_w.append(0)
                    # print('<<<')
                break
            else:
                sourc_back = fr[i + k].strip('\r\n').rstrip('\n').split(' ')[1]
                data_w.append(source_vob[sourc_back])
                # print(sourc_back)

        data_s.append(data_w)
        if len(data_w) is not Poswidth:
            print('____________________', data_w)
        count += 1
        sen_i += 1

    f.close()
    # print(data_t_all)
    return data_s_all

def make_idx_data_index_EE_LSTM3(file, max_s, source_vob):

    data_s_all=[]
    f = open(file,'r')
    fr = f.readlines()
    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        s_sent = sent['tokens']
        data_s = []
        if len(s_sent) > max_s:
            i = 0
            while i < max_s:
                if not source_vob.__contains__(s_sent[i]):
                    data_s.append(source_vob["UNK"])
                else:
                    data_s.append(source_vob[s_sent[i]])
                i += 1
        else:
            i = 0
            while i < len(s_sent):
                if not source_vob.__contains__(s_sent[i]):
                    data_s.append(source_vob["UNK"])
                else:
                    data_s.append(source_vob[s_sent[i]])
                i += 1
            num = max_s - len(s_sent)
            for inum in range(0, num):
                data_s.append(0)

        data_s_all.append(data_s)

    f.close()
    return data_s_all

def get_word_index(train, test):

    source_vob = {}
    target_vob = {}
    sourc_idex_word = {}
    target_idex_word = {}
    count = 1
    tarcount=1

    max_s = 0
    num=0
    token =0

    if not target_vob.__contains__("O"):
        target_vob["O"] = tarcount
        target_idex_word[tarcount] = "O"
        tarcount += 1

    f = open(train,'r')
    fr = f.readlines()
    for line in fr:
        if line.__len__() <= 1:
            if num > max_s:
                max_s = num
            # print(max_s, '  ', num)
            num = 0
            continue
        token +=1

        num += 1
        sourc = line.strip('\r\n').rstrip('\n').split(' ')
        # print(sourc)
        if not source_vob.__contains__(sourc[0]):
            source_vob[sourc[0]] = count
            sourc_idex_word[count] = sourc[0]
            count += 1

        if not target_vob.__contains__(sourc[4]):
            target_vob[sourc[4]] = tarcount
            target_idex_word[tarcount] = sourc[4]
            tarcount += 1
    f.close()

    print('token', token)
    num = 0
    for testf in test:
        f = open(testf, 'r')
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
        count+=1

    target_idex_word = {1: 'O', 2: 'I', 3: 'B', 4: 'E', 5: 'S'}
    target_vob = {'O': 1, 'I': 2, 'B': 3, 'E': 4, 'S': 5}

    return source_vob, sourc_idex_word, target_vob, target_idex_word, max_s


def get_Feature_index(file):
    """
    Give each feature labelling an index
    :param the entlabelingfile file
    :return: the word_index map, the index_word map,
    the max lenth of word sentence
    """
    label_vob = {}
    label_idex_word = {}
    count = 1
    # count = 0

    for labelingfile in file:
        f = open(labelingfile, 'r')
        fr = f.readlines()
        for line in fr:
            if line.__len__() <= 1:

                continue

            sourc = line.strip('\r\n').rstrip('\n').split(' ')[1]
            # print(sourc)
            if not label_vob.__contains__(sourc):
                label_vob[sourc] = count
                label_idex_word[count] = sourc
                count += 1

        f.close()
    if not label_vob.__contains__("**UNK**"):
        label_vob["**UNK**"] = count
        label_idex_word[count] = "**UNK**"
        count += 1


    return label_vob, label_idex_word


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
            # if sourc.__len__() > max_c:
            #     max_c = sourc.__len__()
            #     print(sourc)

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


def get_data(trainfile, devfile, testfile, w2v_file, c2v_file, datafile, w2v_k=300, c2v_k=25, maxlen = 50):
    """
    数据处理的入口函数
    """
    source_vob, sourc_idex_word, target_vob, target_idex_word, max_s = get_word_index(trainfile, {devfile, testfile})
    print("source vocab size: ", str(len(source_vob)))
    print("target vocab size: ", str(len(target_vob)))
    print("target vocab size: " + str(target_vob))
    print("target vocab size: " + str(target_idex_word))
    print ('max soure sent lenth is ' + str(max_s))

    source_w2v, source_W= load_vec_txt(w2v_file,source_vob,  k=w2v_k)
    print("word2vec loaded!")
    print("all vocab size: " + str(len(source_vob)))
    print("source_W  size: " + str(len(source_W)))
    print ("num words in source word2vec: " + str(len(source_w2v)))

    train = make_idx_data_index(trainfile,max_s,source_vob,target_vob)
    dev = make_idx_data_index(devfile,max_s,source_vob,target_vob)
    test = make_idx_data_index(testfile, max_s, source_vob, target_vob)


    source_char, sourc_idex_char, max_c = get_Character_index({trainfile, devfile, testfile})
    print("source char size: ", source_char.__len__())
    print("max_c: ", max_c)
    print("source char: " + str(sourc_idex_char))

    character_W = load_vec_character(c2v_file, source_char,c2v_k)
    print('character_W shape:',character_W.shape)

    chartrain = make_idx_character_index(trainfile,max_s, max_c, source_char)
    chardev = make_idx_character_index(devfile, max_s, max_c, source_char)
    chartest = make_idx_character_index(testfile, max_s, max_c, source_char)

    print(datafile, "dataset created!")
    out = open(datafile, 'wb')#
    pickle.dump([train, dev, test, chartrain, chardev, chartest,
                 source_W, character_W,
                 source_vob, sourc_idex_word, target_vob, target_idex_word, source_char,
                 max_s, w2v_k, max_c, c2v_k], out, 0)
    out.close()


if __name__=="__main__":
    print(20*2)

    alpha = 10
    maxlen = 50
    trainfile = "./data/CoNLL2003_NER/eng.train.BIOES.txt"
    devfile = "./data/CoNLL2003_NER/eng.testa.BIOES.txt"
    testfile = "./data/CoNLL2003_NER/eng.testb.BIOES.txt"
    w2v_file = "./data/w2v/glove.6B.300d.txt"
    datafile = "./data/model/data.pkl"
    modelfile = "./data/model/model.pkl"
    resultdir = "./data/result/"




