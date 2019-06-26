# -*- encoding:utf-8 -*-

# import tensorflow as tf
# config = tf.ConfigProto(allow_soft_placement=True)
# #最多占gpu资源的70%
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# #开始不会给tensorflow全部gpu资源 而是按需增加
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

import pickle, datetime, codecs, math
import os.path
import numpy as np
from PrecessData_Siamese import get_data
from Evaluate import evaluation_NER
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from network.NN_Siamese import Model_BiLSTM__MLP, Model_BiLSTM__MLP_context


def Train41stsegment(sen2list_test):

    model1name = 'Model_BiLSTM_CRF'

    print(model1name)

    w2v_file = "./data/w2v/glove.6B.100d.txt"
    c2v_file = "./data/w2v/C0NLL2003.NER.c2v.txt"
    # trainfile = "./data/CoNLL2003_NER/eng.train.BIOES.txt"
    # devfile = "./data/CoNLL2003_NER/eng.testa.BIOES.txt"
    trainfile = "./data/CoNLL2003_NER/eng.train.testa.BIOES.txt"
    testfile = "./data/CoNLL2003_NER/eng.testb.BIOES.txt"
    resultdir = "./data/result/"


    datafname = 'data_2step.BIOES.A_B.1'
    datafile = "./model_data/" + datafname + ".pkl"

    model1file = 'model1file'

    batch_size = 50
    Test = True

    if not os.path.exists(datafile):
        print("Precess data....")
        from ProcessData_BIOES2F import get_data_4segment_BIOES
        get_data_4segment_BIOES(trainfile, testfile, w2v_file, c2v_file, datafile, w2v_k=100, c2v_k=50, maxlen=maxlen)

    print('Loading data ...')
    train_A_4segment_BIOES, train_B_4segment_BIOES, test_4segment_BIOES, \
    word_W, character_W, \
    word_vob, word_idex_word, char_vob, \
    max_s, w2v_k, max_c, c2v_k = pickle.load(open(datafile, 'rb'))
    print("data has extisted: " + datafile)

    target1_idex_word = {1: 'O', 2: 'I', 3: 'B', 4: 'E', 5: 'S'}
    target1_vob = {'O': 1, 'I': 2, 'B': 3, 'E': 4, 'S': 5}

    trainx_word = np.asarray(train_A_4segment_BIOES[0], dtype="int32")
    trainx_char = np.asarray(train_A_4segment_BIOES[2], dtype="int32")
    trainy = np.asarray(train_A_4segment_BIOES[1], dtype="int32")

    testx_word = np.asarray(test_4segment_BIOES[0], dtype="int32")
    testx_char = np.asarray(test_4segment_BIOES[2], dtype="int32")
    testy = np.asarray(test_4segment_BIOES[1], dtype="int32")

    import TrainModel_segment
    model_segment = TrainModel_segment.SelectModel(modelname=model1name,
                                                   sourcevocabsize=len(word_vob),
                                                   targetvocabsize=len(target1_vob),
                                                   source_W=word_W,
                                                   input_seq_lenth=max_s,
                                                   output_seq_lenth=max_s,
                                                   hidden_dim=200,
                                                   emd_dim=w2v_k,
                                                   sourcecharsize=len(char_vob),
                                                   character_W=character_W,
                                                   input_word_length=max_c,
                                                   char_emd_dim=c2v_k,
                                                   batch_size=batch_size)

    if os.path.exists(model1file):
        model_segment.load_weights(model1file)

    inputs_train_x = [trainx_word, trainx_char]
    inputs_train_y = [trainy]
    inputs_test_x = [testx_word, testx_char]
    inputs_test_y = [testy]
    print("model_segment has extisted...")


    inum = 1
    model1file = "./model/" + model1name + "__" + datafname + "_segment_" + str(inum) + ".h5"

    if Test:
        print("test model_segment ....")
        print(datafile)
        print(model1file)
        TrainModel_segment.infer_e2e_model(model=model_segment, modelname=model1name,
                                           modelfile=model1file,
                                           inputs=[inputs_test_x, inputs_test_y],
                                           target_idex_word=target1_idex_word,
                                           resultdir=resultdir,
                                           batch_size=batch_size)

    fragment_test, target_right = get_data_fromBIOES_2Test(model_segment, test_4segment_BIOES, target1_idex_word, sen2list_test)

    return fragment_test, target_right


def get_data_fromBIOES_2Test(nn_model, test_4segment_BIOES, index2BIOES, sen2list_test):

    index2BIOES[0] = ''

    testx_word = np.asarray(test_4segment_BIOES[0], dtype="int32")
    testx_char = np.asarray(test_4segment_BIOES[2], dtype="int32")
    testy = np.asarray(test_4segment_BIOES[1], dtype="int32")
    testt = test_4segment_BIOES[3]

    predictions = nn_model.predict([testx_word, testx_char], batch_size=512, verbose=1)

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

    fragment_list, target_right = Lists2Set_2Test(ptag_BIOES_all, test_4segment_BIOES[0], testt, sen2list_test)

    print('len(fragment_list) = ', len(fragment_list))
    print('the count right target is ', target_right)

    return fragment_list, target_right

def Lists2Set_2Test(ptag_BIOES_all, testx_word, testt, sen2list_test):
    reall_right = 0
    predict = 0
    fragment_list = []
    fragtuples_list = []
    print('start processing ptag_BIOES_all ...')
    for id, ptag2list in enumerate(ptag_BIOES_all):

        assert len(ptag2list) == len(testt[id])
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
                        tuple = (target_left, target_right, reltag, sen2list_test[id])
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
                tuple = (target_left, target_right, reltag, sen2list_test[id])
                fragtuples_list.append(tuple)
                index += 1

            else:
                index += 1

    P = reall_right / predict
    R = reall_right / 5648.0
    F = 2 * P * R / (P + R)
    print('Lists2Set_42ndTest----', 'P=', P, 'R=', R, 'F=', F)

    return fragtuples_list, reall_right


def test_model_withBIOES(nn_model, fragment_test, target_vob, max_s, max_posi, max_fragment):

    predict = 0
    predict_right = 0
    totel_right = 5648

    data_s_all = []
    data_posi_all = []
    data_tag_all = []
    data_context_r_all = []
    data_context_l_all = []
    data_fragment_all = []
    data_c_l_posi_all = []
    data_c_r_posi_all = []

    fragment_tag_list = []
    for frag in fragment_test:
        fragment_l = int(frag[0])
        fragment_r = int(frag[1])
        if frag[2] == 'NULL':
            fragment_tag = len(target_vob)
        else:
            fragment_tag = target_vob[frag[2]]
        sent = frag[3]

        fragment_tag_list.append(fragment_tag)

        data_s = sent[0:min(len(sent), max_s)] + [0] * max(0, max_s - len(sent))
        data_context_r = sent[fragment_l:min(len(sent), max_s)]
        data_context_r = data_context_r + [0] * max(0, max_s - len(data_context_r))
        data_context_l = sent[max(0, fragment_r - max_s):fragment_r]
        data_context_l = [0] * max(0, max_s - len(data_context_l)) + data_context_l

        data_fragment = sent[fragment_l:fragment_r]

        list_left = [min(i, max_posi) for i in range(1, fragment_l+1)]
        list_left.reverse()
        feature_posi = list_left + [0 for i in range(fragment_l, fragment_r)] + \
                       [min(i, max_posi) for i in range(1, len(sent) - fragment_r + 1)]
        data_posi = feature_posi[0:min(len(sent), max_s)] + [max_posi] * max(0, max_s - len(sent))

        data_c_l_posi = [min(i, max_posi) for i in range(1, len(data_context_l)-len(data_fragment)+1)]
        data_c_l_posi.reverse()
        data_c_l_posi = data_c_l_posi + [0 for i in range(fragment_l, fragment_r+1)]
        data_c_r_posi = [0 for i in range(fragment_l, fragment_r + 1)] + \
                        [min(i, max_posi) for i in range(1, len(data_context_r) - len(data_fragment) + 1)]

        padlen = max(0, max_fragment - len(data_fragment))
        data_fragment = [0] * (padlen // 2) + data_fragment + [0] * (padlen - padlen // 2)

        data_context_r = [1] + data_context_r
        data_context_l = data_context_l + [1]

        for ins in target_vob.values():
            data_s_all.append(data_s)
            data_posi_all.append(data_posi)
            data_tag_all.append([ins])
            data_context_l_all.append(data_context_l)
            data_context_r_all.append(data_context_r)
            data_fragment_all.append(data_fragment)
            data_c_l_posi_all.append(data_c_l_posi)
            data_c_r_posi_all.append(data_c_r_posi)

    pairs = [data_s_all, data_posi_all, data_tag_all,
             data_context_r_all, data_context_l_all, data_fragment_all, data_c_l_posi_all, data_c_r_posi_all]

    x1_sent = np.asarray(pairs[0], dtype="int32")
    x1_posi = np.asarray(pairs[1], dtype="int32")
    x2_tag = np.asarray(pairs[2], dtype="int32")
    x1_context_r = np.asarray(pairs[3], dtype="int32")
    x1_context_l = np.asarray(pairs[4], dtype="int32")
    x1_fragment = np.asarray(pairs[5], dtype="int32")
    x1_c_l_posi = np.asarray(pairs[6], dtype="int32")
    x1_c_r_posi = np.asarray(pairs[7], dtype="int32")


    # predictions = nn_model.predict([x1_sent, x1_posi, x2_tag], batch_size=512, verbose=0)
    predictions = nn_model.predict([x1_context_l, x1_c_l_posi,
                                    x1_context_r, x1_c_r_posi,
                                    x1_fragment, x2_tag], batch_size=512, verbose=0)
    Ddict = {}
    Vdict = {}
    assert len(predictions)//len(target_vob) == len(fragment_tag_list)
    for i in range(len(predictions)//len(target_vob)):
        subpredictions = predictions[i*len(target_vob):i*len(target_vob) + len(target_vob)]
        subpredictions = subpredictions.flatten().tolist()

        u = 0.25 * (subpredictions[0] + subpredictions[1] + subpredictions[2] + subpredictions[3])
        D = 0
        for v in subpredictions:
            D += math.pow(v-u, 2)
        D = 0.25 * D

        mindis = min(subpredictions)
        mindis_where = subpredictions.index(min(subpredictions))

        # for num, disvlaue in enumerate(predictions):
        #     if disvlaue < mindis:
        #         mindis = disvlaue
        #         mindis_where = pairs[2][num]
        if mindis < 0.5:
            predict += 1
            if mindis_where == fragment_tag_list[i]:
                predict_right += 1

                if mindis//0.01 not in Ddict:
                    Ddict[mindis//0.01] = 1
                else:
                    Ddict[mindis//0.01] += 1

                # if D//0.001 not in Ddict:
                #     Ddict[D//0.001] = 1
                # else:
                #     Ddict[D//0.001] += 1

                # if mindis // 0.01 not in Vdict:
                #     Vdict[mindis // 0.01] = 1
                # else:
                #     Vdict[mindis // 0.01] += 1

        if len(target_vob) == fragment_tag_list[i]:
            if mindis//0.01 not in Vdict:
                Vdict[mindis//0.01] = 1
            else:
                Vdict[mindis//0.01] += 1






    P = predict_right / predict
    R = predict_right / totel_right
    F = 2 * P * R / (P + R)
    print('predict_right =, predict =, totel_right = ', predict_right, predict, totel_right)
    print('P = ', P, 'R = ', R, 'F = ', F)

    # Dlist = sorted(Ddict.items(), key=lambda x:x[0], reverse=True)
    # print(Dlist)
    # Vlist = sorted(Vdict.items(), key=lambda x:x[0], reverse=True)
    # print(Vlist)

    return P, R, F


def test_model(nn_model, fragment_test, target_vob, max_s, max_posi, max_fragment):

    predict = 0
    predict_right = 0
    totel_right = len(fragment_test)

    data_s_all = []
    data_posi_all = []
    data_tag_all = []
    data_context_r_all = []
    data_context_l_all = []
    data_fragment_all = []
    data_c_l_posi_all = []
    data_c_r_posi_all = []

    fragment_tag_list = []
    for frag in fragment_test:
        fragment_l = int(frag[0])
        fragment_r = int(frag[1])
        fragment_tag = target_vob[frag[2]]
        sent = frag[3]

        fragment_tag_list.append(fragment_tag)

        data_s = sent[0:min(len(sent), max_s)] + [0] * max(0, max_s - len(sent))
        data_context_r = sent[fragment_l:min(len(sent), max_s)]
        data_context_r = data_context_r + [0] * max(0, max_s - len(data_context_r))
        data_context_l = sent[max(0, fragment_r - max_s):fragment_r]
        data_context_l = [0] * max(0, max_s - len(data_context_l)) + data_context_l

        data_fragment = sent[fragment_l:fragment_r]

        list_left = [min(i, max_posi) for i in range(1, fragment_l+1)]
        list_left.reverse()
        feature_posi = list_left + [0 for i in range(fragment_l, fragment_r)] + \
                       [min(i, max_posi) for i in range(1, len(sent) - fragment_r + 1)]
        data_posi = feature_posi[0:min(len(sent), max_s)] + [max_posi] * max(0, max_s - len(sent))

        data_c_l_posi = [min(i, max_posi) for i in range(1, len(data_context_l)-len(data_fragment)+1)]
        data_c_l_posi.reverse()
        data_c_l_posi = data_c_l_posi + [0 for i in range(fragment_l, fragment_r+1)]
        data_c_r_posi = [0 for i in range(fragment_l, fragment_r + 1)] + \
                        [min(i, max_posi) for i in range(1, len(data_context_r) - len(data_fragment) + 1)]

        padlen = max(0, max_fragment - len(data_fragment))
        data_fragment = [0] * (padlen // 2) + data_fragment + [0] * (padlen - padlen // 2)

        data_context_r = [1] + data_context_r
        data_context_l = data_context_l + [1]

        for ins in target_vob.values():
            data_s_all.append(data_s)
            data_posi_all.append(data_posi)
            data_tag_all.append([ins])
            data_context_l_all.append(data_context_l)
            data_context_r_all.append(data_context_r)
            data_fragment_all.append(data_fragment)
            data_c_l_posi_all.append(data_c_l_posi)
            data_c_r_posi_all.append(data_c_r_posi)

    pairs = [data_s_all, data_posi_all, data_tag_all,
             data_context_r_all, data_context_l_all, data_fragment_all, data_c_l_posi_all, data_c_r_posi_all]

    x1_sent = np.asarray(pairs[0], dtype="int32")
    x1_posi = np.asarray(pairs[1], dtype="int32")
    x2_tag = np.asarray(pairs[2], dtype="int32")
    x1_context_r = np.asarray(pairs[3], dtype="int32")
    x1_context_l = np.asarray(pairs[4], dtype="int32")
    x1_fragment = np.asarray(pairs[5], dtype="int32")
    x1_c_l_posi = np.asarray(pairs[6], dtype="int32")
    x1_c_r_posi = np.asarray(pairs[7], dtype="int32")


    # predictions = nn_model.predict([x1_sent, x1_posi, x2_tag], batch_size=512, verbose=0)
    predictions = nn_model.predict([x1_context_l, x1_c_l_posi,
                                    x1_context_r, x1_c_r_posi,
                                    x1_fragment, x2_tag], batch_size=512, verbose=0)

    assert len(predictions)//len(target_vob) == len(fragment_tag_list)
    for i in range(len(predictions)//len(target_vob)):
        subpredictions = predictions[i*len(target_vob):i*len(target_vob) + len(target_vob)]
        subpredictions = subpredictions.flatten().tolist()

        mindis = min(subpredictions)
        mindis_where = subpredictions.index(min(subpredictions))

        # mincount = 0
        # for num, disvlaue in enumerate(subpredictions):
        #     if disvlaue < 0.5:
        #         mindis_where = num
        #         mincount += 1
        # if mincount == 1:
        #     predict += 1
        #     if mindis_where == fragment_tag_list[i]:
        #         predict_right += 1

        if mindis < 0.5:
            predict += 1
            if mindis_where == fragment_tag_list[i]:
                predict_right += 1

    P = predict_right / predict
    R = predict_right / totel_right
    F = 2 * P * R / (P + R)
    print('predict_right =, predict =, totel_right = ', predict_right, predict, totel_right)

    return P, R, F


def test_model_tagging(nn_model, testdata, chardata, index2type, test_target_count):

    testx_fragment = np.asarray(testdata[0], dtype="int32")
    testx_leftcontext = np.asarray(testdata[1], dtype="int32")
    testx_rightcontext = np.asarray(testdata[2], dtype="int32")
    testy = np.asarray(testdata[3], dtype="int32")
    testchar_fragment = np.asarray(chardata[0], dtype="int32")
    testchar_leftcontext = np.asarray(chardata[1], dtype="int32")
    testchar_rightcontext = np.asarray(chardata[2], dtype="int32")


    predictions = nn_model.predict([testx_fragment, testx_leftcontext, testx_rightcontext,
                                   testchar_fragment, testchar_leftcontext, testchar_rightcontext],
                                   batch_size=512,
                                      verbose=0)

    predict_right = 0
    predict = 0
    target = test_target_count

    for num, ptagindex in enumerate(predictions):

        next_index = np.argmax(ptagindex)
        ptag = index2type[next_index]

        if ptag != 'NULL':
            predict += 1

        if not any(testy[num]):
            ttag = 'NULL'

        else:
            ttag = index2type[np.argmax(testy[num])]

        if ptag == ttag and ttag != 'NULL':
            predict_right += 1

    P = predict_right / predict
    R = predict_right / target
    F = 2 * P * R / (P + R)
    print('predict_right =, predict =, target =, len(predictions) =', predict_right, predict, target, len(predictions))
    print('P= ', P)
    print('R= ', R)
    print('F= ', F)

    return P, R, F


def train_e2e_model(nn_model, modelfile, inputs_train_x, inputs_train_y,
                    inputs_dev_x, inputs_dev_y, fragment_train, fragment_dev, fragment_test,
                    resultdir, type_vob, max_s, max_posi, max_fragment,
                    npoches=100, batch_size=50, retrain=False):

    if retrain:
        nn_model.load_weights(modelfile)
        modelfile = modelfile + '.2nd.h5'

    nn_model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=8)
    checkpointer = ModelCheckpoint(filepath=modelfile + ".best_model.h5", monitor='val_loss', verbose=0,
                                   save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, min_lr=0.00001)

    # nn_model.fit(x_word, y,
    #              batch_size=batch_size,
    #              epochs=npochos,
    #              verbose=1,
    #              shuffle=True,
    #              # validation_split=0.2,
    #              validation_data=(x_word_val, y_val),
    #              callbacks=[reduce_lr, checkpointer, early_stopping])
    #
    # save_model(nn_model, modelfile)
    # # nn_model.save(modelfile, overwrite=True)

    nowepoch = 1
    increment = 1
    earlystop = 0
    maxF = 0.
    while nowepoch <= npoches:
        nowepoch += increment
        earlystop += 1

        nn_model.fit(inputs_train_x, inputs_train_y,
                               batch_size=batch_size,
                               epochs=increment,
                               validation_data=(inputs_dev_x, inputs_dev_y),
                               shuffle=True,
                               class_weight=None,
                               verbose=1,
                               callbacks=[reduce_lr, checkpointer])

        print('the train result-----------------------')
        P, R, F = test_model(nn_model, fragment_train, type_vob, max_s, max_posi, max_fragment)
        print(P, R, F)
        print('the dev result-----------------------')
        P, R, F = test_model(nn_model, fragment_dev, type_vob, max_s, max_posi, max_fragment)
        print(P, R, F)
        print('the test result-----------------------')
        P, R, F = test_model(nn_model, fragment_test, type_vob, max_s, max_posi, max_fragment)

        if F > maxF:
            earlystop = 0
            maxF = F
            nn_model.save_weights(modelfile, overwrite=True)

        print(nowepoch, P, R, F, '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>maxF=', maxF)

        if earlystop >= 50:
            break

    return nn_model


def infer_e2e_model(nnmodel, modelname, modelfile, fragment_test, resultdir, type_vob, max_s, max_posi, max_fragment):

    nnmodel.load_weights(modelfile)
    resultfile = resultdir + "result-" + modelname + '-' + str(datetime.datetime.now())+'.txt'

    print('the train result-----------------------')
    P, R, F = test_model(nn_model, fragment_train, type_vob, max_s, max_posi, max_fragment)
    print(P, R, F)
    print('the dev result-----------------------')
    P, R, F = test_model(nn_model, fragment_dev, type_vob, max_s, max_posi, max_fragment)
    print(P, R, F)
    print('the test result-----------------------')
    P, R, F = test_model(nnmodel, fragment_test, type_vob, max_s, max_posi, max_fragment)
    print('P = ', P, 'R = ', R, 'F = ', F)


def SelectModel(modelname, wordvocabsize, tagvocabsize, posivocabsize,
                     word_W, posi_W, tag_W,
                     input_sent_lenth, input_frament_lenth,
                     w2v_k, posi2v_k, tag2v_k,
                     batch_size=32):
    nn_model = None

    if modelname is 'Model_BiLSTM__MLP':
        nn_model = Model_BiLSTM__MLP(wordvocabsize=wordvocabsize,
                                     tagvocabsize=tagvocabsize,
                                     posivocabsize=posivocabsize,
                                     word_W=word_W, posi_W=posi_W, tag_W=tag_W,
                                     input_sent_lenth=input_sent_lenth,
                                     w2v_k=w2v_k, posi2v_k=posi2v_k, tag2v_k=tag2v_k,
                                     batch_size=batch_size)

    if modelname is 'Model_BiLSTM__MLP_context':
        nn_model = Model_BiLSTM__MLP_context(wordvocabsize=wordvocabsize,
                                             tagvocabsize=tagvocabsize,
                                             posivocabsize=posivocabsize,
                                             word_W=word_W, posi_W=posi_W, tag_W=tag_W,
                                             input_sent_lenth=input_sent_lenth,
                                             input_frament_lenth=input_frament_lenth,
                                             w2v_k=w2v_k, posi2v_k=posi2v_k, tag2v_k=tag2v_k,
                                             batch_size=batch_size)

    return nn_model



if __name__ == "__main__":

    maxlen = 50

    hasNeg = True

    modelname = 'Model_BiLSTM__MLP'
    # modelname = 'Model_BiLSTM__MLP_attention'
    # modelname = 'Model_BiLSTM__MLP_attention'
    modelname = 'Model_BiLSTM__MLP_context'

    print(modelname)

    w2v_file = "./data/w2v/glove.6B.100d.txt"
    c2v_file = "./data/w2v/C0NLL2003.NER.c2v.txt"
    trainfile = "./data/CoNLL2003_NER/eng.train.BIOES.txt"
    devfile = "./data/CoNLL2003_NER/eng.testa.BIOES.txt"
    testfile = "./data/CoNLL2003_NER/eng.testb.BIOES.txt"
    resultdir = "./data/result/"

    datafname = 'data_Siamese.4_allneg_segmentNeg'#1,3, 4_allneg
    datafile = "./model_data/" + datafname + ".pkl"

    modelfile = "next ...."

    batch_size = 512 #256
    hidden_dim = 200
    SecondTrain = True
    retrain = False
    Test = True
    Test42Step = False

    if not os.path.exists(datafile):
        print("Precess data....")

        get_data(trainfile, devfile, testfile, w2v_file, c2v_file, datafile,
                 w2v_k=100, c2v_k=50, maxlen=maxlen, hasNeg=True)

    pairs_train, labels_train, pairs_dev, labels_dev, \
    fragment_train, fragment_dev, fragment_test,\
    word_vob, word_id2word, word_W, w2v_k,\
    TYPE_vob, TYPE_id2type, type_W, type_k,\
    posi_W, posi_W,\
    target_vob, target_id2word, max_s, max_posi, max_fragment = pickle.load(open(datafile, 'rb'))

    train_x1_sent = np.asarray(pairs_train[0], dtype="int32")
    train_x1_posi = np.asarray(pairs_train[1], dtype="int32")
    train_x2_tag = np.asarray(pairs_train[2], dtype="int32")
    train_x1_context_r = np.asarray(pairs_train[3], dtype="int32")
    train_x1_context_l = np.asarray(pairs_train[4], dtype="int32")
    train_x1_fragment = np.asarray(pairs_train[5], dtype="int32")
    train_x1_c_l_posi = np.asarray(pairs_train[6], dtype="int32")
    train_x1_c_r_posi = np.asarray(pairs_train[7], dtype="int32")
    train_y = np.asarray(labels_train, dtype="int32")

    dev_x1_sent = np.asarray(pairs_dev[0], dtype="int32")
    dev_x1_posi = np.asarray(pairs_dev[1], dtype="int32")
    dev_x2_tag = np.asarray(pairs_dev[2], dtype="int32")
    dev_x1_context_r = np.asarray(pairs_dev[3], dtype="int32")
    dev_x1_context_l = np.asarray(pairs_dev[4], dtype="int32")
    dev_x1_fragment = np.asarray(pairs_dev[5], dtype="int32")
    dev_x1_c_l_posi = np.asarray(pairs_dev[6], dtype="int32")
    dev_x1_c_r_posi = np.asarray(pairs_dev[7], dtype="int32")
    dev_y = np.asarray(labels_dev, dtype="int32")

    nn_model = SelectModel(modelname,
                           wordvocabsize=len(word_vob),
                           tagvocabsize=len(TYPE_vob),
                           posivocabsize=max_posi+1,
                           word_W=word_W, posi_W=posi_W, tag_W=type_W,
                           input_sent_lenth=max_s, input_frament_lenth=max_fragment,
                           w2v_k=w2v_k, posi2v_k=max_posi+1, tag2v_k=type_k,
                           batch_size=batch_size)

    # inputs_train_x = [train_x1_sent, train_x1_posi, train_x2_tag]
    inputs_train_x = [train_x1_context_l, train_x1_c_l_posi,
                      train_x1_context_r, train_x1_c_r_posi,
                      train_x1_fragment, train_x2_tag]
    inputs_train_y = [train_y]

    # inputs_dev_x = [dev_x1_sent, dev_x1_posi, dev_x2_tag]
    inputs_dev_x = [dev_x1_context_l, dev_x1_c_l_posi,
                    dev_x1_context_r, dev_x1_c_r_posi,
                    dev_x1_fragment, dev_x2_tag]
    inputs_dev_y = [dev_y]


    for inum in range(6, 9):

        modelfile = "./model/" + modelname + "__" + datafname + "_tagging_" + str(inum) + ".h5"

        if not os.path.exists(modelfile):
            print("Lstm data has extisted: " + datafile)
            print("Training EE model....")
            print(modelfile)
            train_e2e_model(nn_model, modelfile, inputs_train_x, inputs_train_y,
                    inputs_dev_x, inputs_dev_y, fragment_train, fragment_dev, fragment_test,
                    resultdir, TYPE_vob, max_s, max_posi, max_fragment, npoches=100, batch_size=50, retrain=False)

        else:
            if retrain:
                print("ReTraining EE model....")
                train_e2e_model(nn_model, modelfile, inputs_train_x, inputs_train_y,
                                inputs_dev_x, inputs_dev_y, fragment_train, fragment_dev, fragment_test,
                                resultdir, TYPE_vob, max_s, max_posi, max_fragment, npoches=100, batch_size=50, retrain=False)

        if Test:
            print("test EE model....")
            print(datafile)
            print(modelfile)
            infer_e2e_model(nn_model, modelname, modelfile, fragment_test, resultdir, TYPE_vob, max_s, max_posi, max_fragment)

        if Test42Step:
            print("Test42Step ---------------------------------------")
            from PrecessData_Siamese import ReadfromTXT2Lists
            sen2list_test, tag2list_test = ReadfromTXT2Lists(testfile, word_vob, target_vob)
            print('sen2list_train len = ', len(sen2list_test))
            print('tag2list_all len = ', len(tag2list_test))

            fragment_test2, target_right = Train41stsegment(sen2list_test)
            nn_model.load_weights(modelfile)
            test_model_withBIOES(nn_model, fragment_test2, TYPE_vob, max_s, max_posi, max_fragment)



# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
#
# KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))

# CUDA_VISIBLE_DEVICES=1 python3 TrainModel.py

