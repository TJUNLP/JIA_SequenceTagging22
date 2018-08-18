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
    return w2v,k,W

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


def load_vec_character(vocab_c_inx, k=30):

    W = np.zeros(shape=(vocab_c_inx.__len__()+1, k))

    for i in vocab_c_inx:
        W[vocab_c_inx[i]] = np.random.uniform(-1*math.sqrt(3/k), math.sqrt(3/k), k)

    return W,k


def load_vec_onehot(vocab_w_inx):
    """
    Loads 300x1 word vecs from word2vec
    """
    k=vocab_w_inx.__len__()

    W = np.zeros(shape=(vocab_w_inx.__len__()+1, k+1))


    for word in vocab_w_inx:
        W[vocab_w_inx[word],vocab_w_inx[word]] = 1.
    # W[1, 1] = 1.
    return k, W

def make_idx_data_index(file, max_s, source_vob, target_vob):

    data_s_all=[]
    data_t_all=[]
    data_tO_all = []
    data_tBIOES_all = []
    f = open(file,'r')
    fr = f.readlines()

    count = 0
    data_t = []
    data_tO = []
    data_tBIOES = []
    data_s = []
    for line in fr:

        if line.__len__() <= 1:
            num = max_s - count
            # print('num ', num, 'max_s', max_s, 'count', count)
            for inum in range(0, num):
                data_s.append(0)
                targetvec = np.zeros(len(target_vob) + 1)
                targetvec[0] = 1
                data_t.append(targetvec)

                targetvecO = np.zeros(2 + 1)
                targetvecO[0] = 1
                data_tO.append(targetvecO)

                targetvecBIOES = np.zeros(5 + 1)
                targetvecBIOES[0] = 1
                data_tBIOES.append(targetvecBIOES)

            # print(data_s)
            # print(data_t)
            data_s_all.append(data_s)
            data_t_all.append(data_t)
            data_tO_all.append(data_tO)
            data_tBIOES_all.append(data_tBIOES)
            data_t = []
            data_tO = []
            data_tBIOES = []
            data_s = []
            count = 0
            continue

        sent = line.strip('\r\n').rstrip('\n').split(' ')
        if not source_vob.__contains__(sent[0]):
            data_s.append(source_vob["**UNK**"])
        else:
            data_s.append(source_vob[sent[0]])

        # data_t.append(target_vob[sent[4]])
        targetvec = np.zeros(len(target_vob) + 1)
        targetvec[target_vob[sent[4]]] = 1
        data_t.append(targetvec)

        targetvecO = np.zeros(2 + 1)
        if sent[4] == 'O':
            targetvecO[1] = 1
        else:
            targetvecO[2] = 1
        data_tO.append(targetvecO)

        targetvecBIOES = np.zeros(5 + 1)
        if sent[4] == 'O':
            targetvecBIOES[3] = 1
        elif sent[4][0] == 'B':
            targetvecBIOES[1] = 1
        elif sent[4][0] == 'I':
            targetvecBIOES[2] = 1
        elif sent[4][0] == 'E':
            targetvecBIOES[4] = 1
        elif sent[4][0] == 'S':
            targetvecBIOES[5] = 1
        data_tBIOES.append(targetvecBIOES)

        count += 1


    f.close()
    return [data_s_all, data_t_all, data_tO_all, data_tBIOES_all]


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

def make_idx_character_index_withFix(file, max_s, max_c, source_vob):

    data_s_all = []
    count = 0
    f = open(file,'r')
    fr = f.readlines()

    # prefixs = []
    # prefix = open("./data/EnFix/EnPrefix.txt", 'r')
    # pf = prefix.readlines()
    # for line in pf:
    #     prefixs.append(line)

    suffixs = []
    suffix = open("./data/EnFix/EnSuffix.txt", 'r')
    sf = suffix.readlines()
    for line in sf:
        suffixs.append(line)

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
            count = 0
            continue

        data_c = []
        word = line.strip('\r\n').rstrip('\n').split(' ')[0]
        count_c = 0
        start = 0
        end = 0
        # print(word)
        # for pre in prefixs:
        #     pre = pre.strip('\r\n').rstrip('\n').rstrip('\r')
        #     if re.match(pre, word, flags=re.I) is not None:
        #         character = pre
        #         data_c.append(source_vob[character])
        #         start = character.__len__()
        #         count_c +=1
        #         break

        endidex = 0
        for suf in suffixs:
            suf = suf.strip('\r\n').rstrip('\n').rstrip('\r')
            if re.search(suf + '$', word, flags=re.I) is not None:

                character = suf
                endidex=source_vob[character]
                end = character.__len__()

                break

        for chr in range(start, min(word.__len__() - end, max_c)):
            count_c += 1
            if not source_vob.__contains__(word[chr]):
                data_c.append(source_vob["**UNK**"])
            else:
                data_c.append(source_vob[word[chr]])
        if count_c < max_c:
            data_c.append(endidex)
            count_c += 1

        num = max_c - count_c
        for i in range(0, max(num, 0)):
            data_c.append(0)
        count +=1
        # print(data_c.__len__())
        # print(data_c)
        if data_c.__len__() != max_c:
            while 1>0:
                i=1
        data_w.append(data_c)

    f.close()
    return data_s_all


def make_idx_POS_index(file, max_s, source_vob):

    count = 0
    data_s_all = []
    data_s = []
    strat_sen = True
    f = open(file,'r')
    fr = f.readlines()
    for i, line in enumerate(fr):

        if line.__len__() <= 1:
            num = max_s - count
            # print('num ', num, 'max_s', max_s, 'count', count)
            for inum in range(0, num):
                data_s.append([0, 0, 0])
            # print(data_s)
            # print(data_t)
            data_s_all.append(data_s)
            data_s = []
            count = 0
            strat_sen = True
            continue

        data_w = []

        if strat_sen is True:
            data_w.append(0)
        else:
            sourc_pre = fr[i - 1].strip('\r\n').rstrip('\n').split(' ')[1]
            data_w.append(source_vob[sourc_pre])

        sent = line.strip('\r\n').rstrip('\n').split(' ')[1]
        if not source_vob.__contains__(sent):
            data_w.append(source_vob["**UNK**"])
        else:
            data_w.append(source_vob[sent])

        if i + 1 >= fr.__len__() or fr[i + 1].__len__() <= 1:
            data_w.append(0)
        else:
            sourc_back = fr[i + 1].strip('\r\n').rstrip('\n').split(' ')[1]
            data_w.append(source_vob[sourc_back])

        data_s.append(data_w)

        count += 1
        strat_sen = False

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
    if not source_vob.__contains__("**END**"):
        source_vob["**END**"] = count
        sourc_idex_word[count] = "**END**"
        count+=1
    if not source_vob.__contains__("**UNK**"):
        source_vob["**UNK**"] = count
        sourc_idex_word[count] = "**UNK**"
        count+=1
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



def get_Character_index_withFix(files):


    source_vob = {}
    sourc_idex_word = {}
    max_c = 18-5
    count = 1

    # prefixs =[]
    # prefix = open("./data/EnFix/EnPrefix.txt", 'r')
    # pf = prefix.readlines()
    # for line in pf:
    #     prefixs.append(line)

    suffixs = []
    suffix = open("./data/EnFix/EnSuffix.txt", 'r')
    sf = suffix.readlines()
    for line in sf:
        suffixs.append(line)

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
            start = 0
            end = 0
            # for pre in prefixs:
            #     pre = pre.strip('\r\n').rstrip('\n').rstrip('\r')
            #     if re.match(pre, sourc, flags=re.I) is not None:
            #         # print('1@@@@@@@@@@@@@', pre)
            #         character = pre
            #         if not source_vob.__contains__(character):
            #             source_vob[character] = count
            #             sourc_idex_word[count] = character
            #             count += 1
            #             start = character.__len__()
            #         break

            for suf in suffixs:
                suf = suf.strip('\r\n').rstrip('\n').rstrip('\r')
                if re.search(suf + '$', sourc, flags=re.I) is not None:
                    # print('2#############', suf)
                    character = suf
                    if not source_vob.__contains__(character):
                        source_vob[character] = count
                        sourc_idex_word[count] = character
                        count += 1
                        end = character.__len__()
                    break
            # t = sourc.__len__()
            # if end is not 0:
            #     t = sourc.__len__() - end + 1
            # if t > max_c:
            #     max_c = t
            #     print('max_c', max_c, sourc)

            for i in range(start, sourc.__len__()-end):
                character = sourc[i]
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







def get_data(trainfile,devfile, testfile,w2v_file,datafile,w2v_k=300,char_emd_dim=25, maxlen = 50):
    """
    数据处理的入口函数
    Converts the input files  into the end2end model input formats
    :param the train tag file: produced by TaggingScheme.py
    :param the test tag file: produced by TaggingScheme.py
    :param the word2vec file: Extracted form the word2vec resource
    :param: the maximum sentence length we want to set
    :return: tthe end2end model formats data: eelstmfile
    """
    source_vob, sourc_idex_word, target_vob, target_idex_word, max_s = get_word_index(trainfile, {devfile, testfile})

    print("source vocab size: ", str(len(source_vob)))
    print("target vocab size: ", str(len(target_vob)))
    print("target vocab size: " + str(target_vob))
    print("target vocab size: " + str(target_idex_word))

    source_w2v ,k ,source_W= load_vec_txt(w2v_file,source_vob,k=w2v_k)
    # source_w2v, k, source_W, source_vob = load_vec_txt_all(w2v_file,source_vob)

    print("word2vec loaded!")
    print("all vocab size: " + str(len(source_vob)))
    print("source_W  size: " + str(len(source_W)))
    print ("num words in source word2vec: " + str(len(source_w2v))+\
          " source  unknown words: "+str(len(source_vob)-len(source_w2v)))

    # if max_s > maxlen:
    #     max_s = maxlen

    print ('max soure sent lenth is ' + str(max_s))

    train = make_idx_data_index(trainfile,max_s,source_vob,target_vob)
    dev = make_idx_data_index(devfile,max_s,source_vob,target_vob)
    test = make_idx_data_index(testfile, max_s, source_vob, target_vob)

    pos_vob, pos_idex_word = get_Feature_index([trainfile,devfile,testfile])
    pos_train = make_idx_POS_index(trainfile, max_s, pos_vob)
    pos_dev = make_idx_POS_index(devfile, max_s, pos_vob)
    pos_test = make_idx_POS_index(testfile, max_s, pos_vob)
    pos_W, pos_k = load_vec_character(pos_vob, 30)
    # pos_k, pos_W = load_vec_onehot(pos_vob)

    # print('entlabel vocab size:'+str(len(entlabel_vob)))
    print('shape in pos_W:', pos_W.shape)

    withFix = True

    if withFix is True:
        source_char, sourc_idex_char, max_c = get_Character_index_withFix({trainfile, devfile, testfile})
    else:
        source_char, sourc_idex_char, max_c = get_Character_index({trainfile, devfile, testfile})

    print("source char size: ", source_char.__len__())
    print("max_c: ", max_c)
    print("source char: " + str(sourc_idex_char))

    character_W, character_k = load_vec_character(source_char,char_emd_dim)
    print('character_W shape:',character_W.shape)

    if withFix is True:
        chartrain = make_idx_character_index_withFix(trainfile,max_s, max_c, source_char)
        chardev = make_idx_character_index_withFix(devfile, max_s, max_c, source_char)
        chartest = make_idx_character_index_withFix(testfile, max_s, max_c, source_char)
    else:
        chartrain = make_idx_character_index(trainfile,max_s, max_c, source_char)
        chardev = make_idx_character_index(devfile, max_s, max_c, source_char)
        chartest = make_idx_character_index(testfile, max_s, max_c, source_char)

    print ("dataset created!")
    out = open(datafile,'wb')
    pickle.dump([train, dev, test, source_W, source_vob, sourc_idex_word,
                target_vob, target_idex_word, max_s, k,
                 chartrain,chardev,chartest, source_char, character_W, max_c, char_emd_dim,
                 pos_train, pos_dev, pos_test, pos_vob, pos_idex_word, pos_W, pos_k], out)
    out.close()

def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)

def peplacedigital(s):
    if len(s)==1:
        s='1'
    elif len(s)==2:
        s='10'
    elif len(s)==3:
        s='100'
    else:
        s='1000'
    return s

def getClass_weight(x=10):
    cw = {0: 1, 1: 1}
    for i in range(2, x+1):
        cw[i] = 10
    return cw

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
    #
    # source_vob, sourc_idex_word, target_vob, target_idex_word, max_s = get_word_index(trainfile, {testfile})
    #
    #
    #
    # pos_vob, pos_idex_word = get_Feature_index([trainfile,devfile,testfile])
    # pos_train = make_idx_data_index_EE_LSTM2(trainfile, max_s, pos_vob)
    # pos_dev = make_idx_data_index_EE_LSTM2(devfile, max_s, pos_vob)
    # pos_test = make_idx_data_index_EE_LSTM2(testfile, max_s, pos_vob)
    #
    # pos_k, pos_W = load_vec_onehot(pos_vob)
    # print(pos_vob)
    # print(pos_test)
    # print('entlabel vocab size:'+str(len(pos_vob)))
    # print('shape in onhotvec:',pos_W.shape)


    #
    # source_vob, sourc_idex_word, target_vob, target_idex_word, max_s = get_word_index(trainfile, {testfile})
    #
    # print("source vocab size: " + str(len(source_vob)))
    # print("target vocab size: " + str(len(target_vob)))
    # print("target vocab size: " + str(target_vob))
    # print("target vocab size: " + str(target_idex_word))
    #
    # source_w2v ,k ,source_W= load_vec_txt(w2v_file,source_vob)
    # # source_w2v, k, source_W, source_vob = load_vec_txt_all(w2v_file,source_vob)
    #
    # print("word2vec loaded!")
    # print("all vocab size: " + str(len(source_vob)))
    # print("source_W  size: " + str(len(source_W)))
    # print ("num words in source word2vec: " + str(len(source_w2v))+\
    #       " source  unknown words: "+str(len(source_vob)-len(source_w2v)))

    # # if max_s > maxlen:
    # #     max_s = maxlen
    #
    # print ('max soure sent lenth is ' + str(max_s))
    #
    # train = make_idx_data_index(trainfile,max_s,source_vob,target_vob)
    # test = make_idx_data_index(testfile, max_s, source_vob, target_vob)
    #
    # source_char, sourc_idex_char, max_c = get_Character_index({trainfile, devfile, testfile})
    #
    # print("source char size: ", source_char.__len__())
    # print("max_c: ", max_c)
    # print("source char: " + str(sourc_idex_char))
    #
    # character_W = load_vec_character(source_char)
    # print('character_W shape:',character_W.shape)
    #
    # chardata = make_idx_character_index(testfile, 124, max_c, source_char)
    # for s in chardata:
    #     print(s)


    # max_s=10
    #
    # source_char, sourc_idex_char, max_c =get_Character_index_withFix({trainfile, devfile, testfile})
    #
    # print("source char size: ", source_char.__len__())
    # print("max_c: ", max_c)
    # print("source char: " + str(sourc_idex_char))
    #
    # character_W, character_k = load_vec_character(source_char)
    # print('character_W shape:',character_W.shape)
    #
    #
    # chardev = make_idx_character_index_withFix(devfile, max_s, max_c, source_char)
    # for c in chardev:
    #     for d in c:
    #         print(d)
    #     print('------------------')


    # suffixs = []
    # suffix = open("./data/EnFix/EnSuffix.txt", 'r')
    # sf = suffix.readlines()
    # for line in sf:
    #     suffixs.append(line)
    # for suf in suffixs:
    #     suf = suf.rstrip('\n').rstrip('\r')
    #
    #     sourc = 'brother'
    #     if re.search(suf + '$', sourc, flags=re.I) is not None:
    #         print('2#############')
    #
    # max_s=20
    #
    #
    # pos_vob, pos_idex_word = get_Feature_index([trainfile,devfile,testfile])
    # print('pos_vob',pos_vob)
    #
    # # pos_train = make_idx_POS_index(trainfile, max_s, pos_vob)
    # # pos_dev = make_idx_POS_index(devfile, max_s, pos_vob)
    # pos_test = make_idx_POS_index(testfile, max_s, pos_vob)
    # for w in pos_test:
    #     print("....", w)





