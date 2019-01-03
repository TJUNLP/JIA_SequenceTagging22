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
import matplotlib.pyplot as plt
from PrecessData_syntaxaware import get_data
from Evaluate import evaluation_NER_syntaxaware
from network.NN_single import Model_BiLSTM_CRF, Model_BiLSTM_CnnDecoder



def test_model(nn_model, testdata, chardata, pos_data, index2word, resultfile='', batch_size=50):
    index2word[0] = ''

    testx = np.asarray(testdata[0], dtype="int32")
    testy = np.asarray(testdata[1], dtype="int32")
    testpos = np.asarray(pos_data, dtype="int32")
    testchar = np.asarray(chardata, dtype="int32")

    testresult = []

    predictions = nn_model.predict([testx, testchar])

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

    P, R, F, PR_count, P_count, TR_count = evaluation_NER_syntaxaware(testresult, resultfile)

    return P, R, F, PR_count, P_count, TR_count


def train_e2e_model(Modelname, datafile, modelfile, resultdir, npochos=100,hidden_dim=200, batch_size=50, retrain=False):
    # load training data and test data

    traindata, devdata, testdata, source_W, source_vob, sourc_idex_word, \
    target_vob, target_idex_word, max_s, k, \
    chartrain, chardev, chartest, source_char, character_W, max_c, char_emd_dim, \
            pos_train, pos_dev, pos_test, pos_vob, pos_idex_word, pos_W, pos_k \
                = pickle.load(open(datafile, 'rb'))

    # train model
    x_word = np.asarray(traindata[0], dtype="int32")
    y = np.asarray(traindata[1], dtype="int32")
    # y_O = np.asarray(traindata[2], dtype="int32")
    # y_BIOES = np.asarray(traindata[3], dtype="int32")
    # y_Type = np.asarray(traindata[4], dtype="int32")
    # entlabel_train = np.asarray(entlabel_traindata, dtype="int32")
    # poslabel_train = np.asarray(poslabel_traindata, dtype="int32")
    input_char = np.asarray(chartrain, dtype="int32")
    # input_pos = np.asarray(pos_train, dtype="int32")
    x_word_val = np.asarray(devdata[0], dtype="int32")
    y_val = np.asarray(devdata[1], dtype="int32")
    # y_O_val = np.asarray(devdata[2], dtype="int32")
    # y_BIOES_val = np.asarray(devdata[3], dtype="int32")
    # y_Type_val = np.asarray(devdata[4], dtype="int32")

    input_char_val = np.asarray(chardev, dtype="int32")
    # input_pos_val = np.asarray(pos_dev, dtype="int32")

    nn_model = SelectModel(Modelname, sourcevocabsize=len(source_vob), targetvocabsize=len(target_vob),
                                     source_W=source_W,
                                     input_seq_lenth=max_s,
                                     output_seq_lenth=max_s,
                                     hidden_dim=hidden_dim, emd_dim=k,
                           sourcecharsize=len(source_char),
                           character_W=character_W,
                           input_word_length=max_c, char_emd_dim=char_emd_dim,
                           sourcepossize=len(pos_vob),pos_W=pos_W,pos_emd_dim=pos_k)

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
        # for x_word, y, x_word_val, y_val, input_char, input_char_val,x_pos_train, x_pos_dev,sample_weight \
        #         in get_training_xy_otherset(i, x_train, y_train,
        #                                           x_dev, y_dev,
        #                                           max_s,max_c,
        #                                           chartrain, chardev,
        #                                           pos_train, pos_dev,
        #                                           len(target_vob), target_idex_word,
        #                                     sample_weight_value=30,
        #                                     shuffle=True):
        #, y_BIOES, y_Type   , y_BIOES_val, y_Type_val
        history = nn_model.fit([x_word, input_char], [y],
                               batch_size=batch_size,
                               epochs=1,
                               validation_data=([x_word_val, input_char_val], [y_val]),
                               shuffle=True,
                               # sample_weight =sample_weight,
                               verbose=1)

        # plt.plot(history.history['acc'])
        # plt.plot(history.history['val_acc'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
        # # summarize history for loss
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.show()

        if epoch >= saveepoch:
        # if epoch >=0:
            saveepoch += save_inter
            resultfile = ''
            print('the dev result-----------------------')
            P, R, F, PR_count, P_count, TR_count = test_model(nn_model, devdata, chardev, pos_dev, target_idex_word, resultfile, batch_size)
            print(P, R, F)
            print('the test result-----------------------')
            P, R, F, PR_count, P_count, TR_count = test_model(nn_model, testdata, chartest, pos_test, target_idex_word, resultfile,
                                                          batch_size)

            if F > maxF:
                earlystopping = 0
                maxF=F
                nn_model.save_weights(modelfile, overwrite=True)

            else:
                earlystopping += 1

            print(epoch, P, R, F, '  maxF=', maxF)

        if earlystopping >= 10:
            break

    return nn_model


def infer_e2e_model(modelname, datafile, lstm_modelfile, resultdir, hidden_dim=200, batch_size=50):

    traindata, devdata, testdata, source_W, source_vob, sourc_idex_word, \
    target_vob, target_idex_word, max_s, k, \
    chartrain, chardev, chartest, source_char, character_W, max_c, char_emd_dim, \
            pos_train, pos_dev, pos_test, pos_vob, pos_idex_word, pos_W, pos_k \
                = pickle.load(open(datafile, 'rb'))

    nnmodel = SelectModel(modelname, sourcevocabsize=len(source_vob), targetvocabsize=len(target_vob),
                                     source_W=source_W,
                                     input_seq_lenth=max_s,
                                     output_seq_lenth=max_s,
                                     hidden_dim=hidden_dim, emd_dim=k,
                           sourcecharsize=len(source_char),
                           character_W=character_W,
                           input_word_length=max_c, char_emd_dim=char_emd_dim,
                           sourcepossize=len(pos_vob),pos_W=pos_W, pos_emd_dim=pos_k)

    nnmodel.load_weights(lstm_modelfile)
    # nnmodel = load_model(lstm_modelfile)
    resultfile = resultdir + "result-" + modelname + '-' + str(datetime.datetime.now())+'.txt'

    P, R, F, PR_count, P_count, TR_count = test_model(nnmodel, testdata, chartest,pos_test, target_idex_word, resultfile,
                                                      batch_size)
    print('P= ', P, '  R= ', R, '  F= ', F)


def SelectModel(modelname, sourcevocabsize, targetvocabsize, source_W,
                             input_seq_lenth,
                             output_seq_lenth,
                             hidden_dim, emd_dim,
                     sourcecharsize,character_W,input_word_length,char_emd_dim,
                        sourcepossize, pos_W,pos_emd_dim,
                     loss='categorical_crossentropy', optimizer='rmsprop'):
    nn_model = None
    if modelname is 'Model_BiLSTM_CnnDecoder':
        nn_model = Model_BiLSTM_CnnDecoder(sourcevocabsize=sourcevocabsize, targetvocabsize=targetvocabsize,
                                                       source_W=source_W,
                                                       input_seq_lenth=input_seq_lenth,
                                                       output_seq_lenth=output_seq_lenth,
                                                       hidden_dim=hidden_dim, emd_dim=emd_dim,
                                                        sourcecharsize=sourcecharsize,
                                                        character_W=character_W,
                                                        input_word_length=input_word_length,
                                                        char_emd_dim=char_emd_dim,
                                                 sourcepossize=sourcepossize, pos_W=pos_W, pos_emd_dim=pos_emd_dim)

    elif modelname is 'Model_BiLSTM_CRF':
        nn_model = Model_BiLSTM_CRF(sourcevocabsize=sourcevocabsize, targetvocabsize=targetvocabsize,
                                                       source_W=source_W,
                                                       input_seq_lenth=input_seq_lenth,
                                                       output_seq_lenth=output_seq_lenth,
                                                       hidden_dim=hidden_dim, emd_dim=emd_dim,
                                                        sourcecharsize=sourcecharsize,
                                                        character_W=character_W,
                                                        input_word_length=input_word_length,
                                                        char_emd_dim=char_emd_dim,
                                          sourcepossize=sourcepossize, pos_W=pos_W, pos_emd_dim=pos_emd_dim)


    return nn_model


if __name__ == "__main__":

    maxlen = 50

    # modelname = 'Model_BiLSTM_CnnDecoder'
    modelname = 'Model_BiLSTM_CRF'

    print(modelname)

    w2v_file = "./data/w2v/glove.6B.100d.txt"
    trainfile = "./data/CoNLL2003_NER/eng.train.BIOES.txt.SyntaxAware.txt"
    devfile = "./data/CoNLL2003_NER/eng.testa.BIOES.txt.SyntaxAware.txt"
    testfile = "./data/CoNLL2003_NER/eng.testb.BIOES.txt.SyntaxAware.txt"
    resultdir = "./data/result/"

    withFix = False
    withPos = False

    datafile = "./model/data_fix=" + str(withFix) + "_pos=" + str(withPos) + "_syntaxaware_6.pkl"

    modelfile = "./model/" + modelname + "__" + "data_fix=" + str(withFix) + "_pos=" + str(withPos) + \
                "__SyntaxAware_6_1.h5"


    batch_size = 32
    retrain = False
    Test = True

    if not os.path.exists(datafile):
        print("Precess data....")
        get_data(trainfile,devfile, testfile, w2v_file, datafile, w2v_k=100, char_emd_dim=25, withFix=withFix, maxlen=maxlen)
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

