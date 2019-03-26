#coding=utf-8
__author__ = 'JIA'
import numpy as np
import pickle,random
import json, math
import Seq2fragment
import TrainModel_segment
from Evaluate import evaluation_NER, evaluation_NER2, evaluation_NER_BIOES,evaluation_NER_Type


def get_data_4segment_BIOES(trainfile, testfile, w2v_file, c2v_file, datafile, w2v_k=300, c2v_k=25, maxlen = 50):

    word_vob, word_idex_word, target_vob, target_idex_word, max_s = get_word_index([trainfile, testfile])
    print("source vocab size: " + str(len(word_vob)))
    print("target vocab size: " + str(len(target_vob)))
    print("target vocab size: " + str(target_vob))
    print("target vocab size: " + str(target_idex_word))

    word_W, word_k= load_vec_txt(w2v_file, word_vob, k=w2v_k)
    print("word2vec loaded!")
    print("all vocab size: " + str(len(word_vob)))
    print("source_W  size: " + str(len(word_W)))

    char_vob, char_idex_char, max_c = get_Character_index({trainfile, testfile})
    print("source char size: ", char_vob.__len__())
    print("max_c: ", max_c)
    print("source char: " + str(char_idex_char))

    character_W, character_k = load_vec_character(c2v_file, char_vob, c2v_k)
    print('character_W shape:', character_W.shape)

    target1_idex_word = {1: 'O', 2: 'I', 3: 'B', 4: 'E', 5: 'S'}
    target1_vob = {'O': 1, 'I': 2, 'B': 3, 'E': 4, 'S': 5}

    train_word_all, train_target_all, train_t_all = Data2Index_4segment_BIOES(trainfile, max_s, word_vob, target_vob, target1_vob)
    test_word_all, test_target_all, test_t_all = Data2Index_4segment_BIOES(testfile, max_s, word_vob, target_vob, target1_vob)
    print('train_all size', len(train_word_all), 'target_all', len(train_target_all))
    print('test_all size', len(test_word_all))
    train_char_all = Char2Index_4segment_BIOES(trainfile, max_s, max_c, char_vob)
    test_char_all = Char2Index_4segment_BIOES(testfile, max_s, max_c, char_vob)

    # indices = np.arange(len(train_word_all))
    # np.random.shuffle(indices)
    train_split = int(len(train_word_all) // 2)

    train_A_word = train_word_all[:train_split]
    train_A_target = train_target_all[:train_split]
    train_A_t = train_t_all[:train_split]
    train_B_word = train_word_all[train_split:]
    train_B_target = train_target_all[train_split:]
    train_B_t = train_t_all[train_split:]
    train_A_char = train_char_all[:train_split]
    train_B_char = train_char_all[train_split:]

    train_A_4segment_BIOES = [train_A_word, train_A_target, train_A_char, train_A_t]
    train_B_4segment_BIOES = [train_B_word, train_B_target, train_B_char, train_B_t]
    test_4segment_BIOES = [test_word_all, test_target_all, test_char_all, test_t_all]

    print(datafile, "dataset created!")
    out = open(datafile, 'wb')#
    pickle.dump([train_A_4segment_BIOES, train_B_4segment_BIOES, test_4segment_BIOES,
                 word_W, character_W,
                 word_vob, word_idex_word, char_vob,
                 max_s, w2v_k, max_c, c2v_k], out, 0)
    out.close()


def get_data_4classifer(model_segment, train_B_4segment_BIOES, test_4segment_BIOES, target1_idex_word,
                        max_c, char_vob, word_idex_word, batch_size):

    max_context = 0
    max_fragment = 1
    train_fragment_list, max_context, max_fragment, train_target_right = get_data_42ndTraining(model_segment,
                                                                                   train_B_4segment_BIOES,
                                                                                   max_context, max_fragment,
                                                                                   index2BIOES=target1_idex_word,
                                                                                   batch_size=batch_size, Istest=False)
    test_fragment_list, max_context, max_fragment, test_target_right = get_data_42ndTraining(model_segment,
                                                                                   test_4segment_BIOES,
                                                                                   max_context, max_fragment,
                                                                                   index2BIOES=target1_idex_word,
                                                                                   batch_size=batch_size,
                                                                                   Istest=True)
    test_target_count = 5648

    print('max_context--', max_context, 'max_fragment--', max_fragment)
    print('len(test_fragment_list)---', len(test_fragment_list))
    print('test_target_right--- ', test_target_right)

    Type_idex_word = {0: 'LOC', 1: 'ORG', 2: 'PER', 3: 'MISC'}
    Type_vob = {'LOC': 0, 'ORG': 1, 'PER': 2, 'MISC': 3}

    train = Data2Index_42ndclassifer(train_fragment_list, Type_vob, max_context, max_fragment, hasNeg=False)
    test = Data2Index_42ndclassifer(test_fragment_list, Type_vob, max_context, max_fragment, hasNeg=False)
    print(len(train), len(test))

    chartrain = Char2Index_42ndclassifer(train_fragment_list, max_context, max_fragment, max_c, char_vob, word_idex_word)

    chartest = Char2Index_42ndclassifer(test_fragment_list, max_context, max_fragment, max_c, char_vob, word_idex_word)
    print(len(chartrain), len(chartest))

    print ("dataset created!")

    return train, test, chartrain, chartest, Type_vob, Type_idex_word, max_fragment, max_context, test_target_count




def get_data_4classifer_3l(model_segment, train_B_4segment_BIOES, test_4segment_BIOES, target1_idex_word,
                           max_s, max_c, char_vob, word_idex_word, batch_size):

    max_context = 5
    max_fragment = 1
    train_fragment_list, max_context, max_fragment, train_target_right = get_data_42ndTraining_3l(model_segment,
                                                                                   train_B_4segment_BIOES,
                                                                                   max_context, max_fragment,
                                                                                   index2BIOES=target1_idex_word,
                                                                                   batch_size=batch_size, Istest=False)
    test_fragment_list, max_context, max_fragment, test_target_right = get_data_42ndTraining_3l(model_segment,
                                                                                   test_4segment_BIOES,
                                                                                   max_context, max_fragment,
                                                                                   index2BIOES=target1_idex_word,
                                                                                   batch_size=batch_size,
                                                                                   Istest=True)
    test_target_count = 5648

    print('max_context--', max_context, 'max_fragment--', max_fragment)
    print('len(test_fragment_list)---', len(test_fragment_list))
    print('test_target_right--- ', test_target_right)

    Type_idex_word = {0: 'LOC', 1: 'ORG', 2: 'PER', 3: 'MISC'}
    Type_vob = {'LOC': 0, 'ORG': 1, 'PER': 2, 'MISC': 3}

    train = Data2Index_42ndclassifer_3l(train_fragment_list, Type_vob, max_context, max_fragment, hasNeg=False)
    test = Data2Index_42ndclassifer_3l(test_fragment_list, Type_vob, max_context, max_fragment, hasNeg=False)
    print(len(train), len(test))

    chartrain = Char2Index_42ndclassifer(train_fragment_list, max_context, max_fragment, max_c, char_vob, word_idex_word)

    chartest = Char2Index_42ndclassifer(test_fragment_list, max_context, max_fragment, max_c, char_vob, word_idex_word)
    print(len(chartrain), len(chartest))

    feature_posi_k, feature_posi_W = load_vec_onehot(k=max_s)
    print('feature_posi_k, feature_posi_W', feature_posi_k, len(feature_posi_W))

    print ("dataset created!")

    return train, test, chartrain, chartest, Type_vob, Type_idex_word, \
           feature_posi_k, feature_posi_W, \
           max_fragment, max_context, test_target_count


def load_vec_onehot(k=124):


    vocab_w_inx = [i for i in range(0, k)]

    W = np.zeros(shape=(vocab_w_inx.__len__(), k))


    for word in vocab_w_inx:
        W[vocab_w_inx[word],vocab_w_inx[word]] = 1.

    return k, W


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


def Char2Index_42ndclassifer(fraglist, max_context, max_fragment, max_c, char_vob, word_idex_word):

    char_fragment_all = []
    char_leftcontext_all = []
    char_rightcontext_all = []

    for line in fraglist:

        fragment = line[0]
        context_left = line[2]
        context_right = line[3]

        data_fragment = []
        for wordindex in fragment[0 : min(len(fragment), max_fragment)]:

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


def Char2Index_4segment_BIOES(file, max_s, max_c, source_vob):

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
        count += 1
        data_w.append(data_c)

    f.close()
    return data_s_all


def Data2Index_4segment_BIOES(file, max_s, source_vob, target_vob, target1_vob):

    data_s_all=[]
    data_tBIOES_all = []
    data_t_all = []

    f = open(file, 'r')
    fr = f.readlines()

    count = 0

    data_tBIOES = []
    data_t = []
    data_s = []
    for line in fr:

        if line.__len__() <= 1:

            data_s = data_s + [0] * max(0, max_s - count)
            data_tBIOES = data_tBIOES + [[1] + [0] * 5] * max(0, max_s - count)

            data_s_all.append(data_s)
            data_t_all.append(data_t)
            data_tBIOES_all.append(data_tBIOES)

            data_tBIOES = []
            data_t = []
            data_s = []
            count = 0
            continue

        sent = line.strip('\r\n').rstrip('\n').split(' ')
        if not source_vob.__contains__(sent[0]):
            data_s.append(source_vob["**UNK**"])
        else:
            data_s.append(source_vob[sent[0]])


        targetvecBIOES = np.zeros(len(target1_vob) + 1)
        targetvecBIOES[target1_vob[sent[4][0]]] = 1
        data_tBIOES.append(targetvecBIOES)

        # targetvec = np.zeros(len(target_vob) + 1)
        # targetvec[target_vob[sent[4]]] = 1
        data_t.append(sent[4])

        count += 1

    f.close()
    return data_s_all, data_tBIOES_all, data_t_all


def Data2Index_42ndclassifer(fraglist, word2index_Type, max_context, max_fragment, hasNeg=True):

    data_fragment_all = []
    data_leftcontext_all = []
    data_rightcontext_all = []
    data_t_all = []
    data_t_2tpye_all = []

    for line in fraglist:

        fragment = line[0]
        fragment_tag = line[1]
        context_left = line[2]
        context_right = line[3]

        data_fragment = fragment[0:min(len(fragment), max_fragment)] + [0] * max(0, max_fragment-len(fragment))
        if hasNeg:
            data_t = np.zeros(5)
        else:
            data_t = np.zeros(4)

        if fragment_tag in word2index_Type.keys():
            data_t[word2index_Type[fragment_tag]] = 1
        data_t_2tp = np.zeros(2)
        if fragment_tag == 'NULL':
            data_t_2tp[0] = 1
        else:
            data_t_2tp[1] = 1

        data_leftcontext = [0] * max(0, max_context-len(context_left)) + context_left
        data_rightcontext = context_right + [0] * max(0, max_context-len(context_right))

        data_fragment_all.append(data_fragment)
        data_leftcontext_all.append(data_leftcontext)
        data_rightcontext_all.append(data_rightcontext)
        data_t_all.append(data_t)
        data_t_2tpye_all.append(data_t_2tp)

    return [data_fragment_all, data_leftcontext_all, data_rightcontext_all, data_t_all, data_t_2tpye_all]


def Data2Index_42ndclassifer_3l(fraglist, word2index_Type, max_context, max_fragment, hasNeg=True):

    data_fragment_all = []
    data_leftcontext_all = []
    data_rightcontext_all = []
    data_feature_posi_all = []
    data_sentence_all = []
    data_t_all = []
    data_t_2tpye_all = []

    for line in fraglist:

        fragment = line[0]
        fragment_tag = line[1]
        context_left = line[2]
        context_right = line[3]
        feature_posi = line[4]
        feature_sent = line[5]

        data_fragment = fragment[0:min(len(fragment), max_fragment)] + [0] * max(0, max_fragment-len(fragment))
        if hasNeg:
            data_t = np.zeros(5)
        else:
            data_t = np.zeros(4)

        if fragment_tag in word2index_Type.keys():
            data_t[word2index_Type[fragment_tag]] = 1
        data_t_2tp = np.zeros(2)
        if fragment_tag == 'NULL':
            data_t_2tp[0] = 1
        else:
            data_t_2tp[1] = 1

        data_leftcontext = [0] * max(0, max_context-len(context_left)) + context_left
        data_rightcontext = context_right + [0] * max(0, max_context-len(context_right))



        data_fragment_all.append(data_fragment)
        data_leftcontext_all.append(data_leftcontext)
        data_rightcontext_all.append(data_rightcontext)
        data_feature_posi_all.append(feature_posi)
        data_sentence_all.append(feature_sent)
        data_t_all.append(data_t)
        data_t_2tpye_all.append(data_t_2tp)

    return [data_fragment_all, data_leftcontext_all, data_rightcontext_all, data_t_all,
            data_feature_posi_all, data_sentence_all, data_t_2tpye_all]


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


def test_model_segment(nn_model, testdata, chartest, index2tag):

    index2tag[0] = ''

    testx = np.asarray(testdata[0], dtype="int32")
    testy_BIOES = np.asarray(testdata[1], dtype="int32")
    testchar = np.asarray(chartest, dtype="int32")


    predictions = nn_model.predict([testx, testchar])
    testresult_1Step = []
    testresult2 = []
    for si in range(0, len(predictions)):

        ptag_BIOES = []
        ptag_1Step = []
        for word in predictions[si]:

            next_index = np.argmax(word)
            next_token = index2tag[next_index]
            ptag_BIOES.append(next_token)
            if next_token != '':
                ptag_1Step.append(next_token)

        ttag_BIOES = []
        for word in testy_BIOES[si]:

            next_index = np.argmax(word)
            next_token = index2tag[next_index]
            ttag_BIOES.append(next_token)

        testresult2.append([ptag_BIOES, ttag_BIOES])
        testresult_1Step.append(ptag_1Step)


    P, R, F, PR_count, P_count, TR_count = evaluation_NER_BIOES(testresult2, resultfile='')
    print('divide---BIOES>>>>>>>>>>', P, R, F)

    return testresult_1Step


def get_data_42ndTraining(nn_model, test_4segment_BIOES, max_context, max_fragment, index2BIOES, batch_size=256, Istest=False):

    index2BIOES[0] = ''

    testx_word = np.asarray(test_4segment_BIOES[0], dtype="int32")
    testx_char = np.asarray(test_4segment_BIOES[2], dtype="int32")
    testy = np.asarray(test_4segment_BIOES[1], dtype="int32")
    testt = test_4segment_BIOES[3]

    predictions = nn_model.predict([testx_word, testx_char], batch_size=batch_size, verbose=1)

    ptag_BIOES_all = []
    for si in range(0, len(predictions)):

        ptag_BIOES = []
        for word in predictions[si]:
            next_index = np.argmax(word)
            next_token = index2BIOES[next_index]
            if next_token == '':
                break
            ptag_BIOES.append(next_token)

        ptag_BIOES_all.append(ptag_BIOES)


    print('is test ? ', Istest)
    if not Istest:
        fragment_list, max_context, max_fragment, target_right = Lists2Set_42ndTraining(ptag_BIOES_all, test_4segment_BIOES[0], testt, max_context, max_fragment)
    else:
        fragment_list, max_context, max_fragment, target_right = Lists2Set_42ndTest(ptag_BIOES_all, test_4segment_BIOES[0], testt, max_context, max_fragment)

    print('len(fragment_list) = ', len(fragment_list))
    print('the count right target is ', target_right)

    return fragment_list, max_context, max_fragment, target_right



def get_data_42ndTraining_3l(nn_model, test_4segment_BIOES, max_context, max_fragment, index2BIOES, batch_size=256, Istest=False):

    index2BIOES[0] = ''

    testx_word = np.asarray(test_4segment_BIOES[0], dtype="int32")
    testx_char = np.asarray(test_4segment_BIOES[2], dtype="int32")
    testy = np.asarray(test_4segment_BIOES[1], dtype="int32")
    testt = test_4segment_BIOES[3]

    predictions = nn_model.predict([testx_word, testx_char], batch_size=batch_size, verbose=1)

    ptag_BIOES_all = []
    for si in range(0, len(predictions)):

        ptag_BIOES = []
        for word in predictions[si]:
            next_index = np.argmax(word)
            next_token = index2BIOES[next_index]
            if next_token == '':
                break
            ptag_BIOES.append(next_token)

        ptag_BIOES_all.append(ptag_BIOES)


    print('is test ? ', Istest)
    if not Istest:
        fragment_list, max_context, max_fragment, target_right = Lists2Set_42ndTraining_3l(ptag_BIOES_all, test_4segment_BIOES[0], testt, max_context, max_fragment)
    else:
        fragment_list, max_context, max_fragment, target_right = Lists2Set_42ndTest_3l(ptag_BIOES_all, test_4segment_BIOES[0], testt, max_context, max_fragment)

    print('len(fragment_list) = ', len(fragment_list))
    print('the count right target is ', target_right)

    return fragment_list, max_context, max_fragment, target_right


def Lists2Set_42ndTest(ptag_BIOES_all, testx_word, testt, max_context, max_fragment):
    reall_right = 0
    predict = 0
    fragment_list = []


    print('start processing ptag_BIOES_all ...')
    for id, ptag2list in enumerate(ptag_BIOES_all):
        fragtuples_list = []

        if len(ptag2list) != len(testt[id]):
            while (True):

                print('error Lists2Set_42ndTest ....')

        index = 0
        while index < len(ptag2list):
            if ptag2list[index] == 'O' or ptag2list[index] == '':
                index += 1
                continue
            elif ptag2list[index] == 'B':
                target_left = index
                index += 1
                while index < len(ptag2list):
                    if ptag2list[index] == 'I':
                        index += 1
                        continue
                    elif ptag2list[index] == 'E':
                        target_right = index + 1
                        reltag = 'NULL'
                        if 'B-' in testt[id][target_left] and 'E-' in testt[id][index]:
                            reltag = testt[id][target_left][2:]
                            reall_right += 1
                        predict += 1
                        tuple = (0, target_right, target_left, target_right, target_left, len(ptag2list), reltag)
                        fragtuples_list.append(tuple)
                        index += 1
                        break
                    else:
                        break

            elif ptag2list[index] == 'S':
                target_left = index
                target_right = index + 1
                reltag = 'NULL'
                if 'S-' in testt[id][index]:
                    reltag = testt[id][target_left][2:]
                    reall_right += 1
                predict += 1
                tuple = (0, target_right, target_left, target_right, target_left, len(ptag2list), reltag)
                fragtuples_list.append(tuple)
                index += 1
            else:
                index += 1

        for tup in fragtuples_list:
            context_left = testx_word[id][tup[0]:tup[1]]
            fragment = testx_word[id][tup[2]:tup[3]]
            context_right = testx_word[id][tup[4]:tup[5]]
            fragment_tag = tup[6]
            fragment_list.append((fragment, fragment_tag, context_left, context_right))

            max_context = max(max_context, len(context_left), len(context_right))
            max_fragment = max(max_fragment, len(fragment))

    P = reall_right / predict
    R = reall_right / 5648.0
    F = 2 * P * R / (P + R)
    print('Lists2Set_42ndTest----', 'P=', P, 'R=', R, 'F=', F)

    return fragment_list, max_context, max_fragment, reall_right


def Lists2Set_42ndTest_3l(ptag_BIOES_all, testx_word, testt, max_context, max_fragment):
    reall_right = 0
    predict = 0
    fragment_list = []


    print('start processing ptag_BIOES_all ...')
    for id, ptag2list in enumerate(ptag_BIOES_all):
        fragtuples_list = []

        if len(ptag2list) != len(testt[id]):
            while (True):

                print('error Lists2Set_42ndTest ....')

        index = 0
        while index < len(ptag2list):
            if ptag2list[index] == 'O' or ptag2list[index] == '':
                index += 1
                continue
            elif ptag2list[index] == 'B':
                target_left = index
                index += 1
                while index < len(ptag2list):
                    if ptag2list[index] == 'I':
                        index += 1
                        continue
                    elif ptag2list[index] == 'E':
                        target_right = index + 1
                        reltag = 'NULL'
                        if 'B-' in testt[id][target_left] and 'E-' in testt[id][index]:
                            reltag = testt[id][target_left][2:]
                            reall_right += 1
                        predict += 1
                        tuple = (target_left, target_right, len(ptag2list), reltag)
                        fragtuples_list.append(tuple)
                        index += 1
                        break
                    else:
                        break

            elif ptag2list[index] == 'S':
                target_left = index
                target_right = index + 1
                reltag = 'NULL'
                if 'S-' in testt[id][index]:
                    reltag = testt[id][target_left][2:]
                    reall_right += 1
                predict += 1
                tuple = (target_left, target_right, len(ptag2list), reltag)
                fragtuples_list.append(tuple)
                index += 1
            else:
                index += 1

        for tup in fragtuples_list:
            context_left = testx_word[id][max(0, tup[0]-max_context):tup[1]]
            fragment = testx_word[id][tup[0]:tup[1]]
            context_right = testx_word[id][tup[0]:min(tup[2], tup[1]+max_context)]

            list_left = [i for i in range(1, tup[0] + 1)]
            list_left.reverse()
            feature_posi = list_left + [0 for i in range(tup[0], tup[1])] + [i for i in range(1, len(testx_word[id]) - tup[1] + 1)]
            print(tup[0], tup[1])
            print(feature_posi)
            feature_sent = testx_word[id]
            fragment_tag = tup[3]
            fragment_list.append((fragment, fragment_tag, context_left, context_right, feature_posi, feature_sent))

            max_fragment = max(max_fragment, len(fragment))

    P = reall_right / predict
    R = reall_right / 5648.0
    F = 2 * P * R / (P + R)
    print('Lists2Set_42ndTest----', 'P=', P, 'R=', R, 'F=', F)

    return fragment_list, max_context, max_fragment, reall_right


def Lists2Set_42ndTraining(ptag_BIOES_all, testx_word, testt, max_context, max_fragment):
    reall_right = 0
    predict_right = 0
    predict = 0
    fragment_list = []


    print('start processing testt ...')
    for id, tag2list in enumerate(testt):
        fragtuples_list = []
        target_left = 0

        for index, tag in enumerate(tag2list):

            if tag == 'O':
                continue

            else:
                if tag.__contains__('B-'):
                    target_left = index

                elif tag.__contains__('I-'):
                    continue

                else:
                    if tag.__contains__('S-'):

                        target_left = index
                        target_right = index + 1

                    elif tag.__contains__('E-'):

                        target_right = index + 1

                    reltag = tag[2:]
                    tuple_posi = (0, target_right, target_left, target_right, target_left, len(tag2list), reltag)
                    fragtuples_list.append(tuple_posi)
                    reall_right += 1
                    flens = max(index + 1, len(tag2list) - target_left)
                    if flens > max_context:
                        max_context = flens
                    max_fragment = max(max_fragment, target_right - target_left)

        for tup in fragtuples_list:
            context_left = testx_word[id][tup[0]:tup[1]]
            fragment = testx_word[id][tup[2]:tup[3]]
            context_right = testx_word[id][tup[4]:tup[5]]
            fragment_tag = tup[6]
            fragment_list.append((fragment, fragment_tag, context_left, context_right))


    print('start processing ptag_BIOES_all ...')
    for id, ptag2list in enumerate(ptag_BIOES_all):
        fragtuples_list = []
        
        if len(ptag2list)!= len(testt[id]):
            while(True):
                print('error Lists2Set_42ndTraining ....')
        index = 0
        while index < len(ptag2list):

            if ptag2list[index] == 'O' or ptag2list[index] == '':
                index += 1
                continue
            elif ptag2list[index] == 'B':
                target_left = index
                index += 1
                while index < len(ptag2list):
                    if ptag2list[index] == 'I':
                        index += 1
                        continue
                    elif ptag2list[index] == 'E':
                        target_right = index + 1
                        predict += 1
                        if 'B-' in testt[id][target_left] and \
                                'E-' in testt[id][index]:
                            predict_right += 1
                            break

                        tuple = (0, target_right, target_left, target_right, target_left, len(ptag2list), 'NULL')
                        fragtuples_list.append(tuple)
                        index += 1
                        break
                    else:
                        break

            elif ptag2list[index] == 'S':
                predict += 1
                if 'S-' not in testt[id][index]:
                    target_left = index
                    target_right = index + 1
                    tuple = (0, target_right, target_left, target_right, target_left, len(ptag2list), 'NULL')
                    fragtuples_list.append(tuple)
                else:
                    predict_right += 1
                index += 1
            else:
                index += 1

        for tup in fragtuples_list:
            context_left = testx_word[id][tup[0]:tup[1]]
            fragment = testx_word[id][tup[2]:tup[3]]
            context_right = testx_word[id][tup[4]:tup[5]]
            fragment_tag = tup[6]
            fragment_list.append((fragment, fragment_tag, context_left, context_right))

            max_context = max(max_context, len(context_left), len(context_right))
            max_fragment = max(max_fragment, len(fragment))
    

    P = predict_right / predict
    R = predict_right / reall_right
    F = 2 * P * R / (P + R)
    print('Lists2Set_42ndTraining----', 'P=', P, 'R=', R, 'F=', F)


    return fragment_list, max_context, max_fragment, reall_right


def Lists2Set_42ndTraining_3l(ptag_BIOES_all, testx_word, testt, max_context=5, max_fragment=1, ):
    reall_right = 0
    predict_right = 0
    predict = 0
    fragment_list = []

    print('start processing testt ...')
    for id, tag2list in enumerate(testt):
        fragtuples_list = []
        target_left = 0

        for index, tag in enumerate(tag2list):

            if tag == 'O':
                continue

            else:
                if tag.__contains__('B-'):
                    target_left = index

                elif tag.__contains__('I-'):
                    continue

                else:
                    if tag.__contains__('S-'):

                        target_left = index
                        target_right = index + 1

                    elif tag.__contains__('E-'):

                        target_right = index + 1

                    reltag = tag[2:]
                    tuple_posi = (target_left, target_right, len(tag2list), reltag)
                    fragtuples_list.append(tuple_posi)
                    reall_right += 1

                    max_fragment = max(max_fragment, target_right - target_left)

        for tup in fragtuples_list:
            context_left = testx_word[id][max(0, tup[0]-max_context):tup[1]]
            fragment = testx_word[id][tup[0]:tup[1]]
            context_right = testx_word[id][tup[0]:min(tup[2], tup[1]+max_context)]

            list_left = [i for i in range(1, tup[0] + 1)]
            list_left.reverse()
            feature_posi = list_left + [0 for i in range(tup[0], tup[1])] + [i for i in range(1, len(testx_word[id]) - tup[1] + 1)]

            print(feature_posi)
            print(str(tup), len(testx_word[id]))
            feature_sent = testx_word[id]
            fragment_tag = tup[6]
            fragment_list.append((fragment, fragment_tag, context_left, context_right, feature_posi, feature_sent))

    print('start processing ptag_BIOES_all ...')
    for id, ptag2list in enumerate(ptag_BIOES_all):
        fragtuples_list = []

        if len(ptag2list) != len(testt[id]):
            while (True):
                print('error Lists2Set_42ndTraining ....')
        index = 0
        while index < len(ptag2list):

            if ptag2list[index] == 'O' or ptag2list[index] == '':
                index += 1
                continue
            elif ptag2list[index] == 'B':
                target_left = index
                index += 1
                while index < len(ptag2list):
                    if ptag2list[index] == 'I':
                        index += 1
                        continue
                    elif ptag2list[index] == 'E':
                        target_right = index + 1
                        predict += 1
                        if 'B-' in testt[id][target_left] and \
                                'E-' in testt[id][index]:
                            predict_right += 1
                            break

                        tuple = (target_left, target_right, len(ptag2list), 'NULL')
                        fragtuples_list.append(tuple)
                        index += 1
                        break
                    else:
                        break

            elif ptag2list[index] == 'S':
                predict += 1
                if 'S-' not in testt[id][index]:
                    target_left = index
                    target_right = index + 1
                    tuple = (target_left, target_right, len(ptag2list), 'NULL')
                    fragtuples_list.append(tuple)
                else:
                    predict_right += 1
                index += 1
            else:
                index += 1

        for tup in fragtuples_list:
            context_left = testx_word[id][max(0, tup[0]-max_context):tup[1]]
            fragment = testx_word[id][tup[0]:tup[1]]
            context_right = testx_word[id][tup[0]:min(tup[2], tup[1]+max_context)]

            list_left = [i for i in range(1, tup[0] + 1)]
            list_left.reverse()
            feature_posi = list_left + [0 for i in range(tup[0], tup[1])] + [i for i in range(1, len(testx_word[id]) - tup[1] + 1)]
            print(tup[0], tup[1])
            print(feature_posi)
            feature_sent = testx_word[id]
            fragment_tag = tup[3]
            fragment_list.append((fragment, fragment_tag, context_left, context_right, feature_posi, feature_sent))

            max_fragment = max(max_fragment, len(fragment))

    P = predict_right / predict
    R = predict_right / reall_right
    F = 2 * P * R / (P + R)
    print('Lists2Set_42ndTraining----', 'P=', P, 'R=', R, 'F=', F)

    return fragment_list, max_context, max_fragment, reall_right


if __name__ == '__main__':

    resultdir = "./data/result/"




