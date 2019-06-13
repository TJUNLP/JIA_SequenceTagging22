# -*- encoding:utf-8 -*-

import tensorflow as tf
# config = tf.ConfigProto(allow_soft_placement=True)
# #最多占gpu资源的70%
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# #开始不会给tensorflow全部gpu资源 而是按需增加
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

import pickle, datetime, codecs
import os.path
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import PrecessData_multi_nerpos, PrecessData_PreC2V
from Evaluate import evaluation_NER, evaluation_NER2, evaluation_NER_BIOES,evaluation_NER_Type, evaluation_NER_BIOES_TYPE
# from keras.models import Sequential
# from keras.layers.embeddings import Embedding
from keras.layers import Flatten,Lambda,Conv2D
from keras.layers.core import Dropout, Activation, Permute, RepeatVector
from keras.layers.merge import concatenate, Concatenate, multiply, Dot
from keras.layers import TimeDistributed, Input, Bidirectional, Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D, RepeatVector, AveragePooling1D
from keras.models import Model
from keras_contrib.layers import CRF
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
# from keras.losses import my_cross_entropy_withWeight
from network.BiLSTM_CRF_multi3 import BiLSTM_CRF_multi3_with2t_1



def test_model(nn_model, input_x, input_y, index2word, resultfile='', batch_size=50):
    index2word[0] = ''
    index2word_BIOES = {0: '', 1: 'B', 2: 'I', 3: 'O', 4: 'E', 5: 'S'}
    index2word_Type = {0: '', 1: 'O', 2: 'LOC', 3: 'ORG', 4: 'PER', 5: 'MISC'}

    testresult2 = []
    testresult3 = []
    predictions = nn_model.predict(input_x)

    for si in range(0, len(predictions[0])):

        ptag_BIOES = []
        for word in predictions[2][si]:
            next_index = np.argmax(word)
            next_token = index2word_BIOES[next_index]
            ptag_BIOES.append(next_token)
        # print('next_token--ptag--',str(ptag))

        ptag_Type = []
        for word in predictions[1][si]:
            next_index = np.argmax(word)
            next_token = index2word_Type[next_index]
            ptag_Type.append(next_token)
        # print('next_token--ptag--',str(ptag))


        ttag_BIOES = []
        for word in input_y[2][si]:
            next_index = np.argmax(word)
            next_token = index2word_BIOES[next_index]
            ttag_BIOES.append(next_token)

        ttag_Type = []
        for word in input_y[1][si]:
            next_index = np.argmax(word)
            next_token = index2word_Type[next_index]
            ttag_Type.append(next_token)


        result2 = []
        result2.append(ptag_BIOES)
        result2.append(ttag_BIOES)
        testresult2.append(result2)

        result3 = []
        result3.append(ptag_Type)
        result3.append(ttag_Type)
        testresult3.append(result3)


    P, R, F, PR_count, P_count, TR_count = evaluation_NER_BIOES(testresult2, resultfile='')
    print('BIOES>>>>>>>>>>', P, R, F)
    P, R, F, PR_count, P_count, TR_count = evaluation_NER_Type(testresult3, resultfile='')
    print('Type>>>>>>>>>>', P, R, F)

    P, R, F, PR_count, P_count, TR_count = evaluation_NER_BIOES_TYPE(testresult2, testresult3, resultfile='')
    print('Type>>>>>>>>>>', P, R, F)


    return P, R, F, PR_count, P_count, TR_count



def SelectModel(modelname, sourcevocabsize, targetvocabsize, source_W,
                             input_seq_lenth,
                             output_seq_lenth,
                             hidden_dim, emd_dim,
                     sourcecharsize,character_W,input_word_length,char_emd_dim):
    nn_model = None
    if modelname is 'BiLSTM_CRF_multi3_with2t_1':
        nn_model = BiLSTM_CRF_multi3_with2t_1(sourcevocabsize=sourcevocabsize, targetvocabsize=targetvocabsize,
                                                       source_W=source_W,
                                                       input_seq_lenth=input_seq_lenth,
                                                       output_seq_lenth=output_seq_lenth,
                                                       hidden_dim=hidden_dim, emd_dim=emd_dim,
                                                        sourcecharsize=sourcecharsize,
                                                        character_W=character_W,
                                                        input_word_length=input_word_length,
                                                        char_emd_dim=char_emd_dim)


    return nn_model


def train_e2e_model(nn_model, modelfile,
                    input_x, input_y, input_x_val, input_y_val, input_x_test, input_y_test,
                    resultdir, npoches=100, batch_size=50, retrain=False):
    # load training data and test data

    if retrain:
        nn_model.load_weights(modelfile)

    nn_model.summary()


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

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    # val_acc val_crf_viterbi_accuracy
    checkpointer = ModelCheckpoint(filepath=modelfile+".best_model.h5", monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.00001)
    nowepoch = 1
    increment = 1
    earlystop = 0
    maxF = 0.
    while nowepoch <= npoches:
        nowepoch += increment
        earlystop += 1
        nn_model.fit(x=input_x,
                     y=input_y,
                     batch_size=batch_size,
                     epochs=increment,
                     verbose=1,
                     shuffle=True,
                     validation_data=(input_x_val, input_y_val),
                     callbacks=[reduce_lr, checkpointer])

        resultfile = ''
        print('the dev result-----------------------')
        P, R, F, PR_count, P_count, TR_count = test_model(nn_model, input_x_val, input_y_val, target_idex_word,
                                                        resultfile, batch_size)
        print(P, R, F)
        print('the test result-----------------------')
        P, R, F, PR_count, P_count, TR_count = test_model(nn_model, input_x_test, input_y_test, target_idex_word,
                                                          resultfile, batch_size)

        if F > maxF:
            maxF = F
            earlystop = 0
            nn_model.save_weights(modelfile, overwrite=True)

        print(nowepoch, 'P= ', P, '  R= ', R, '  F= ', F, '>>>>>>>>>>>>>>>>>>>>>>>>>>maxF= ', maxF)

        if earlystop > 50:
            break

    return nn_model


def getClass_weight(x=10):
    cw = {0: 1, 1: 1}
    for i in range(2, x+1):
        cw[i] = 10
    return cw

def infer_e2e_model(nnmodel, modelfile, input_x_test, input_y_test, resultdir, batch_size=50):

    nn_model.load_weights(modelfile)
    resultfile = resultdir + "result-" + modelname + '-' + str(datetime.datetime.now())+'.txt'

    P, R, F, PR_count, P_count, TR_count = test_model(nnmodel, input_x_test, input_y_test,
                                                      target_idex_word, resultfile, batch_size)
    print('P= ', P, '  R= ', R, '  F= ', F)


if __name__ == "__main__":

    maxlen = 50

    modelname = 'BiLSTM_CRF_multi3_with2t_1'

    print(modelname)

    resultdir = "./data/result/"
    trainfile = "./data/CoNLL2003_NER/eng.train.BIOES.txt"
    devfile = "./data/CoNLL2003_NER/eng.testa.BIOES.txt"
    testfile = "./data/CoNLL2003_NER/eng.testb.BIOES.txt"
    w2v_file = "./data/w2v/glove.6B.100d.txt"
    c2v_file = "./data/w2v/C0NLL2003.NER.c2v.txt"
    datafname = 'data_multi3_new_1'
    datafile = "./model/" + datafname + ".pkl"
    # modelfile = "./data/model/BiLSTM_CnnDecoder_wordFixCharembed_model3.h5"


    batch_size = 32
    retrain = False
    Test = True
    valid = False
    Label = False
    if not os.path.exists(datafile):
        print("Precess data....")
        PrecessData_PreC2V. \
            get_data(trainfile, devfile, testfile, w2v_file, c2v_file, datafile, w2v_k=100, char_emd_dim=50,
                     withFix=False, maxlen=maxlen)

    train, dev, test, source_W, source_vob, sourc_idex_word, \
    target_vob, target_idex_word, max_s, k, \
    chartrain, chardev, chartest, \
    source_char, character_W, max_c, char_emd_dim = pickle.load(open(datafile, 'rb'))

    # train model
    x_word = np.asarray(train[0], dtype="int32")
    y = np.asarray(train[1], dtype="int32")
    y_O = np.asarray(train[2], dtype="int32")
    y_BIOES = np.asarray(train[3], dtype="int32")
    y_Type = np.asarray(train[4], dtype="int32")

    x_word_val = np.asarray(dev[0], dtype="int32")
    y_val = np.asarray(dev[1], dtype="int32")
    y_O_val = np.asarray(dev[2], dtype="int32")
    y_BIOES_val = np.asarray(dev[3], dtype="int32")
    y_Type_val = np.asarray(dev[4], dtype="int32")

    x_word_test = np.asarray(test[0], dtype="int32")
    y_test = np.asarray(test[1], dtype="int32")
    y_O_test = np.asarray(dev[2], dtype="int32")
    y_BIOES_test = np.asarray(test[3], dtype="int32")
    y_Type_test = np.asarray(test[4], dtype="int32")

    input_char = np.asarray(chartrain, dtype="int32")
    input_char_val = np.asarray(chardev, dtype="int32")
    input_char_test = np.asarray(chartest, dtype="int32")

    inputs_train_x = [x_word, input_char]
    inputs_train_y = [y_O, y_Type, y_BIOES]

    inputs_dev_x = [x_word_val, input_char_val]
    inputs_dev_y = [y_O_val, y_Type_val, y_BIOES_val]

    inputs_test_x = [x_word_test, input_char_test]
    inputs_test_y = [y_O_test, y_Type_test, y_BIOES_test]

    for inum in range(6, 9):

        modelfile = "./model/" + modelname + "__" + datafname + str(inum) + ".h5"

        nn_model = SelectModel(modelname,
                               sourcevocabsize=len(source_vob),
                               targetvocabsize=len(target_vob),
                               source_W=source_W,
                               input_seq_lenth=max_s,
                               output_seq_lenth=max_s,
                               hidden_dim=200,
                               emd_dim=k,
                               sourcecharsize=len(source_char),
                               character_W=character_W,
                               input_word_length=max_c, char_emd_dim=char_emd_dim)

        if not os.path.exists(modelfile):
            print("Lstm data has extisted: " + datafile)
            print("Training EE model....")
            print(modelfile)
            train_e2e_model(nn_model, modelfile,
                            inputs_train_x, inputs_train_y,
                            inputs_dev_x, inputs_dev_y,
                            inputs_test_x, inputs_test_y,
                            resultdir,
                            npoches=100,  batch_size=batch_size, retrain=False)
        else:
            if retrain:
                print("ReTraining EE model....")
                train_e2e_model(nn_model, modelfile,
                            inputs_train_x, inputs_train_y,
                            inputs_dev_x, inputs_dev_y,
                            inputs_test_x, inputs_test_y,
                            resultdir,
                            npoches=100,  batch_size=batch_size, retrain=False)

        if Test:
            print("test EE model....")
            print(modelfile)
            infer_e2e_model(nn_model, modelfile, inputs_test_x, inputs_test_y, resultdir, batch_size=batch_size)


# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
#
# KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))

# CUDA_VISIBLE_DEVICES=1 python3 TrainModel.py

