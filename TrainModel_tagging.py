# -*- encoding:utf-8 -*-

# import tensorflow as tf
# config = tf.ConfigProto(allow_soft_placement=True)
# #最多占gpu资源的70%
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# #开始不会给tensorflow全部gpu资源 而是按需增加
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

import pickle, datetime, codecs
import os.path
import numpy as np
from ProcessData_S2F import get_data
from Evaluate import evaluation_NER

from network.NN_tagging import Model_LSTM_BiLSTM_LSTM



def test_model_segment(nn_model, testdata, chardata, index2word, resultfile='', batch_size=50):

    index2word_Type = {0: '', 1: 'O', 2: 'LOC', 3: 'ORG', 4: 'PER', 5: 'MISC'}
    word2index_Type = {'': 0, 'O': 1, 'LOC': 2, 'ORG': 3, 'PER': 4, 'MISC': 5}

    testx_fragment = np.asarray(testdata[0], dtype="int32")
    testx_leftcontext = np.asarray(testdata[1], dtype="int32")
    testx_rightcontext = np.asarray(testdata[2], dtype="int32")
    testy = np.asarray(testdata[3], dtype="int32")
    testchar_fragment = np.asarray(chardata[0], dtype="int32")
    testchar_leftcontext = np.asarray(chardata[1], dtype="int32")
    testchar_rightcontext = np.asarray(chardata[2], dtype="int32")

    testresult = []

    predictions = nn_model.predict([testx_fragment, testx_leftcontext, testx_rightcontext,
                                    testchar_fragment, testchar_leftcontext, testchar_rightcontext])

    for si in range(0, len(predictions)):

        sent = predictions[si]
        ptag = []
        for word in sent:
            next_index = np.argmax(word)
            next_token = index2word[next_index]
            ptag.append(next_token)
        # print('next_token--ptag--',str(ptag))

        senty = testy[si]
        ttag = []
        for word in senty:
            next_index = np.argmax(word)
            next_token = index2word[next_index]
            ttag.append(next_token)

        result = []
        result.append(ptag)
        result.append(ttag)
        testresult.append(result)

    P, R, F, PR_count, P_count, TR_count = evaluation_NER(testresult, resultfile)

    return P, R, F, PR_count, P_count, TR_count


def train_e2e_model(modelname, datafile, modelfile, resultdir, npochos=100,hidden_dim=200, batch_size=50, retrain=False):
    # load training data and test data

    traindata, devdata, testdata,\
    chartrain, chardev, chartest,\
    word_vob, word_idex_word,\
    Type_vob, Type_idex_word,\
    char_vob, char_idex_char,\
    word_W, word_k,\
    character_W, character_k,\
    max_context, max_fragment, max_c = pickle.load(open(datafile, 'rb'))

    trainx_fragment = np.asarray(traindata[0], dtype="int32")
    trainx_leftcontext = np.asarray(traindata[1], dtype="int32")
    trainx_rightcontext = np.asarray(traindata[2], dtype="int32")
    trainy = np.asarray(traindata[3], dtype="int32")
    trainchar_fragment = np.asarray(chartrain[0], dtype="int32")
    trainchar_leftcontext = np.asarray(chartrain[1], dtype="int32")
    trainchar_rightcontext = np.asarray(chartrain[2], dtype="int32")

    devx_fragment = np.asarray(devdata[0], dtype="int32")
    devx_leftcontext = np.asarray(devdata[1], dtype="int32")
    devx_rightcontext = np.asarray(devdata[2], dtype="int32")
    devy = np.asarray(devdata[3], dtype="int32")
    devchar_fragment = np.asarray(chardev[0], dtype="int32")
    devchar_leftcontext = np.asarray(chardev[1], dtype="int32")
    devchar_rightcontext = np.asarray(chardev[2], dtype="int32")

    testx_fragment = np.asarray(testdata[0], dtype="int32")
    testx_leftcontext = np.asarray(testdata[1], dtype="int32")
    testx_rightcontext = np.asarray(testdata[2], dtype="int32")
    testy = np.asarray(testdata[3], dtype="int32")
    testchar_fragment = np.asarray(chartest[0], dtype="int32")
    testchar_leftcontext = np.asarray(chartest[1], dtype="int32")
    testchar_rightcontext = np.asarray(chartest[2], dtype="int32")

    nn_model = SelectModel(modelname,
                          wordvocabsize=len(word_vob),
                          targetvocabsize=len(Type_vob),
                          charvobsize=len(char_vob),
                          word_W=word_W, char_W=character_W,
                          input_fragment_lenth=max_fragment,
                          input_leftcontext_lenth=max_context,
                          input_rightcontext_lenth=max_context,
                          input_maxword_length=max_c,
                          w2v_k=word_k, c2v_k=character_k,
                          hidden_dim=hidden_dim, batch_size=batch_size)

    if retrain:
        nn_model.load_weights(modelfile)

    nn_model.summary()



    # early_stopping = EarlyStopping(monitor='val_loss', patience=8)
    # checkpointer = ModelCheckpoint(filepath="./data/model/best_model.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0001)
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
    epoch = 0
    save_inter = 1
    saveepoch = save_inter
    maxF = 0
    earlystopping =0
    i = 0
    while (epoch < npochos):
        epoch = epoch + 1
        i += 1
        history = nn_model.fit([trainx_fragment, trainx_leftcontext, trainx_rightcontext,
                                trainchar_fragment, trainchar_leftcontext, trainchar_rightcontext],
                               [trainy],
                               batch_size=batch_size,
                               epochs=1,
                               validation_data=([devx_fragment, devx_leftcontext, devx_rightcontext,
                                                 devchar_fragment, devchar_leftcontext, devchar_rightcontext], [devy]),
                               shuffle=True,
                               # sample_weight =sample_weight,
                               verbose=1)


        if epoch >= saveepoch:
            saveepoch += save_inter
            resultfile = ''

            print('the test result-----------------------')
            loss, acc = nn_model.evaluate([testx_fragment, testx_leftcontext, testx_rightcontext,
                                           testchar_fragment, testchar_leftcontext, testchar_rightcontext],
                                          [testy],
                                          verbose=0,
                                          batch_size=32)

            print('\n test_test score:', loss, acc)

            if acc > maxF:
                earlystopping = 0
                maxF=acc
                nn_model.save_weights(modelfile, overwrite=True)

            else:
                earlystopping += 1

            print(epoch, loss, acc, '  maxF=', maxF)

        if earlystopping >= 10:
            break

    return nn_model


def infer_e2e_model(modelname, datafile, lstm_modelfile, resultdir, hidden_dim=200, batch_size=32):

    traindata, devdata, testdata,\
    chartrain, chardev, chartest,\
    word_vob, word_idex_word,\
    Type_vob, Type_idex_word,\
    char_vob, char_idex_char,\
    word_W, word_k,\
    character_W, character_k,\
    max_context, max_fragment, max_c = pickle.load(open(datafile, 'rb'))

    testx_fragment = np.asarray(testdata[0], dtype="int32")
    testx_leftcontext = np.asarray(testdata[1], dtype="int32")
    testx_rightcontext = np.asarray(testdata[2], dtype="int32")
    testy = np.asarray(testdata[3], dtype="int32")
    testchar_fragment = np.asarray(chartest[0], dtype="int32")
    testchar_leftcontext = np.asarray(chartest[1], dtype="int32")
    testchar_rightcontext = np.asarray(chartest[2], dtype="int32")

    nnmodel = SelectModel(modelname,
                          wordvocabsize=len(word_vob),
                          targetvocabsize=len(Type_vob),
                          charvobsize=len(char_vob),
                          word_W=word_W, char_W=character_W,
                          input_fragment_lenth=max_fragment,
                          input_leftcontext_lenth=max_context,
                          input_rightcontext_lenth=max_context,
                          input_maxword_length=max_c,
                          w2v_k=word_k, c2v_k=character_k,
                          hidden_dim=hidden_dim, batch_size=batch_size)

    nnmodel.load_weights(lstm_modelfile)
    # nnmodel = load_model(lstm_modelfile)

    resultfile = resultdir + "result-" + modelname + '-' + str(datetime.datetime.now())+'.txt'

    loss, acc = nnmodel.evaluate([testx_fragment, testx_leftcontext, testx_rightcontext,
                                   testchar_fragment, testchar_leftcontext, testchar_rightcontext], [testy], verbose=0,
                                  batch_size=10)

    print('\n test_test score:', loss, acc)


def SelectModel(modelname, wordvocabsize, targetvocabsize, charvobsize,
                word_W, char_W,
                input_fragment_lenth, input_leftcontext_lenth,
                input_rightcontext_lenth, input_maxword_length,
                w2v_k, c2v_k,
                hidden_dim=200, batch_size=32):
    nn_model = None

    if modelname is 'Model_LSTM_BiLSTM_LSTM':
        nn_model = Model_LSTM_BiLSTM_LSTM(wordvocabsize=wordvocabsize,
                                          targetvocabsize=targetvocabsize,
                                          charvobsize=charvobsize,
                                          word_W=word_W, char_W=char_W,
                                          input_fragment_lenth=input_fragment_lenth,
                                          input_leftcontext_lenth=input_leftcontext_lenth,
                                          input_rightcontext_lenth=input_rightcontext_lenth,
                                          input_maxword_length=input_maxword_length,
                                          w2v_k=w2v_k, c2v_k=c2v_k,
                                          hidden_dim=hidden_dim, batch_size=batch_size)

    return nn_model


if __name__ == "__main__":

    maxlen = 50

    modelname = 'Model_LSTM_BiLSTM_LSTM'


    print(modelname)

    w2v_file = "./data/w2v/glove.6B.100d.txt"
    c2v_file = "./data/w2v/C0NLL2003.NER.c2v.txt"
    trainfile = "./data/CoNLL2003_NER/eng.train.BIOES.txt"
    devfile = "./data/CoNLL2003_NER/eng.testa.BIOES.txt"
    testfile = "./data/CoNLL2003_NER/eng.testb.BIOES.txt"
    resultdir = "./data/result/"

    datafname = 'data_tagging_4type_PreC2V.1'
    datafile = "./model_data/" + datafname + ".pkl"

    modelfile = "next ...."

    batch_size = 32
    retrain = False
    Test = True

    if not os.path.exists(datafile):
        print("Precess data....")

        get_data(trainfile,devfile, testfile, w2v_file, c2v_file, datafile, w2v_k=100, c2v_k=50)

    for inum in range(0, 3):

        modelfile = "./model/" + modelname + "__" + datafname + "_tagging_" + str(inum) + ".h5"

        if not os.path.exists(modelfile):
            print("Lstm data has extisted: " + datafile)
            print("Training EE model....")
            print(modelfile)
            train_e2e_model(modelname, datafile, modelfile, resultdir,
                            npochos=100, hidden_dim=200, batch_size=batch_size, retrain=False)
        else:
            if retrain:
                print("ReTraining EE model....")
                train_e2e_model(modelname, datafile, modelfile, resultdir,
                                npochos=100, hidden_dim=200, batch_size=batch_size, retrain=retrain)

        if Test:
            print("test EE model....")
            print(datafile)
            print(modelfile)
            infer_e2e_model(modelname, datafile, modelfile, resultdir, hidden_dim=200, batch_size=batch_size)



    # import tensorflow as tf
    # import keras.backend.tensorflow_backend as KTF
    #
    # KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))

    # CUDA_VISIBLE_DEVICES=1 python3 TrainModel.py

