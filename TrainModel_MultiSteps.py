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
import TrainModel_segment
from ProcessData_BIOES2F import get_data_4segment_BIOES, get_data_4classifer, get_data_4classifer_3l
from Evaluate import evaluation_NER
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from network.NN_tagging import Model_LSTM_BiLSTM_LSTM, Model_3Level



def test_model_tagging(nn_model, inputs_test_x, inputs_test_y, index2type, test_target_count):


    predictions = nn_model.predict(inputs_test_x, verbose=1)

    predict_right = 0
    predict = 0
    target = test_target_count

    for num, ptagindex in enumerate(predictions):

        next_index = np.argmax(ptagindex)
        ptag = index2type[next_index]

        if ptag != 'NULL':
            predict += 1

        if not any(inputs_test_y[0][num]):
            ttag = 'NULL'

        else:
            ttag = index2type[np.argmax(inputs_test_y[0][num])]

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
                    inputs_test_x, inputs_test_y,
                    Type_idex_word, test_target_count,
                    resultdir, npochos=100, batch_size=50, retrain=False):
    batch_size = 16

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
    maxF = 0
    earlystopping =0
    i = 0
    epochlen = 1
    while (epoch < npochos):
        epoch = epoch + epochlen
        i += 1
        # if os.path.exists(modelfile):
        #     nn_model.load_weights(modelfile)
        class_weight = {0:5, 1:5, 2:5, 3:5, 4:1}
        checkpointer = ModelCheckpoint(filepath=modelfile + ".best_model.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True)
        history = nn_model.fit(inputs_train_x,
                               inputs_train_y,
                               batch_size=batch_size,
                               epochs=epochlen,
                               validation_split=0.2,
                               shuffle=True,
                               class_weight=class_weight,
                               verbose=1,
                               # callbacks=[checkpointer]
                               )

        print('the test result-----------------------')
        loss, acc = nn_model.evaluate(inputs_test_x, inputs_test_y, verbose=1, batch_size=512)
        print('\n test_test score:', loss, acc)
        P, R, F = test_model_tagging(nn_model, inputs_test_x, inputs_test_y, Type_idex_word, test_target_count)

        # nn_best_model = SelectModel(modelname,
        #               wordvocabsize=len(word_vob),
        #               targetvocabsize=len(Type_vob),
        #               charvobsize=len(char_vob),
        #               word_W=word_W, char_W=character_W,
        #               input_fragment_lenth=max_fragment,
        #               input_leftcontext_lenth=max_context,
        #               input_rightcontext_lenth=max_context,
        #               input_maxword_length=max_c,
        #               w2v_k=word_k, c2v_k=character_k,
        #               hidden_dim=hidden_dim, batch_size=batch_size)
        #
        # nn_best_model.load_weights(modelfile + ".best_model.h5")
        # P_bm, R_bm, F_bm = test_model_tagging(nn_best_model, testdata, chartest, Type_idex_word, test_target_count)

        if F > maxF:
            earlystopping = 0
            maxF = F
            nn_model.save_weights(modelfile, overwrite=True)
        # if F_bm > maxF:
        #     earlystopping = 0
        #     maxF = F_bm
        #     nn_best_model.save_weights(modelfile, overwrite=True)

        else:
            earlystopping += epochlen

        print(epoch, '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>maxF=', maxF)

        if earlystopping >= 10:
            break


    return nn_model


def infer_e2e_model(nnmodel, modelname, modelfile,
                    inputs_test_x, inputs_test_y, Type_idex_word,
                    test_target_count, resultdir, batch_size=32):


    nnmodel.load_weights(modelfile)

    resultfile = resultdir + "result-" + modelname + '-' + str(datetime.datetime.now())+'.txt'

    loss, acc = nnmodel.evaluate(inputs_test_x, inputs_test_y, verbose=0,
                                  batch_size=512)
    print('\n test_test score:', loss, acc)
    test_model_tagging(nnmodel, inputs_test_x, inputs_test_y, Type_idex_word, test_target_count)


def SelectModel(modelname, wordvocabsize, targetvocabsize, charvobsize, posivocabsize,
                word_W, char_W, posi_W,
                input_fragment_lenth, input_leftcontext_lenth,
                input_rightcontext_lenth, input_maxword_length, input_sent_lenth,
                w2v_k, c2v_k, posi_k,
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
    elif modelname is 'Model_3Level':
            nn_model = Model_3Level(wordvocabsize=wordvocabsize,
                                              targetvocabsize=targetvocabsize,
                                              charvobsize=charvobsize,
                                    posivocabsize=posivocabsize,
                                              word_W=word_W, char_W=char_W, posi_W=posi_W,
                                              input_fragment_lenth=input_fragment_lenth,
                                              input_leftcontext_lenth=input_leftcontext_lenth,
                                              input_rightcontext_lenth=input_rightcontext_lenth,
                                              input_maxword_length=input_maxword_length,
                                    input_sent_lenth=input_sent_lenth,
                                              w2v_k=w2v_k, c2v_k=c2v_k, posi_k=posi_k,
                                              hidden_dim=hidden_dim, batch_size=batch_size)

    return nn_model


def Train41stsegment(datafname, datafile, trainfile, testfile, w2v_file, c2v_file, maxlen,
                     model1name, model1file,
                     resultdir, hidden_dim, batch_size, Test):

    if not os.path.exists(datafile):
        print("Precess data....")
        get_data_4segment_BIOES(trainfile, testfile, w2v_file, c2v_file, datafile, w2v_k=100, c2v_k=50, maxlen=maxlen)

    print('Loading data ...')
    train_A_4segment_BIOES, train_B_4segment_BIOES, test_4segment_BIOES,\
    word_W, character_W,\
    word_vob, word_idex_word, char_vob,\
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

    for inum in range(0, 3):

        model1file = "./model/" + model1name + "__" + datafname + "_segment_" + str(inum) + ".h5"

        if not os.path.exists(model1file):

            print("Training model_segment....")
            print(model1file)
            model_segment = TrainModel_segment.train_e2e_model(model=model_segment,
                                               modelfile=model1file,
                                               inputs=[inputs_train_x, inputs_train_y, inputs_test_x, inputs_test_y],
                                               target_idex_word=target1_idex_word,
                                               resultdir=resultdir,
                                               npochos=100, batch_size=batch_size, retrain=False)

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

        Train42ndclassifer(inum, model2name,
                           model_segment, train_B_4segment_BIOES, test_4segment_BIOES,
                           target1_idex_word,
                           word_vob, word_W, character_W, w2v_k, c2v_k,
                           max_s, max_c, char_vob, word_idex_word, resultdir, batch_size)


def Train42ndclassifer(Step_num, model2name,
                       model_segment, train_B_4segment_BIOES, test_4segment_BIOES,
                       target1_idex_word,
                       word_vob, word_W, character_W, word_k, character_k,
                       max_s, max_c, char_vob, word_idex_word, resultdir, batch_size):

    traindata, testdata, chartrain, chartest, \
    Type_vob, Type_idex_word, feature_posi_k, feature_posi_W, \
    max_fragment, max_context, \
    test_target_count = get_data_4classifer_3l(model_segment, train_B_4segment_BIOES, test_4segment_BIOES, target1_idex_word,
                                               max_s, max_c, char_vob, word_idex_word, batch_size)

    trainx_fragment = np.asarray(traindata[0], dtype="int32")
    trainx_leftcontext = np.asarray(traindata[1], dtype="int32")
    trainx_rightcontext = np.asarray(traindata[2], dtype="int32")
    trainy = np.asarray(traindata[3], dtype="int32")
    trainx_posi = np.asarray(traindata[4], dtype="int32")
    trainx_sent = np.asarray(traindata[5], dtype="int32")

    trainchar_fragment = np.asarray(chartrain[0], dtype="int32")
    trainchar_leftcontext = np.asarray(chartrain[1], dtype="int32")
    trainchar_rightcontext = np.asarray(chartrain[2], dtype="int32")

    testx_fragment = np.asarray(testdata[0], dtype="int32")
    testx_leftcontext = np.asarray(testdata[1], dtype="int32")
    testx_rightcontext = np.asarray(testdata[2], dtype="int32")
    testy = np.asarray(testdata[3], dtype="int32")
    testx_posi = np.asarray(testdata[4], dtype="int32")
    testx_sent = np.asarray(testdata[5], dtype="int32")
    testchar_fragment = np.asarray(chartest[0], dtype="int32")
    testchar_leftcontext = np.asarray(chartest[1], dtype="int32")
    testchar_rightcontext = np.asarray(chartest[2], dtype="int32")


    inputs_train_x = [trainx_fragment, trainx_leftcontext, trainx_rightcontext, trainx_posi, trainx_sent,
                    trainchar_fragment, trainchar_leftcontext, trainchar_rightcontext]
    inputs_train_y = [trainy]

    inputs_test_x = [testx_fragment, testx_leftcontext, testx_rightcontext, testx_posi, testx_sent,
                     testchar_fragment, testchar_leftcontext, testchar_rightcontext]
    inputs_test_y = [testy]

    model_classifer = SelectModel(model2name,
                          wordvocabsize=len(word_vob),
                          targetvocabsize=len(Type_vob),
                          charvobsize=len(char_vob),
                          posivocabsize=len(max_s),
                          word_W=word_W, char_W=character_W, posi_W=feature_posi_W,
                          input_fragment_lenth=max_fragment,
                          input_leftcontext_lenth=max_context,
                          input_rightcontext_lenth=max_context,
                          input_maxword_length=max_c,
                          input_sent_lenth=max_s,
                          w2v_k=word_k, c2v_k=character_k, posi_k=feature_posi_k,
                          hidden_dim=200, batch_size=batch_size)

    for inum in range(0, 3):

        model2file = "./model/" + model2name + "_classifer_" + str(Step_num) + '--' + str(inum) + ".h5"

        if not os.path.exists(model2file):

            print("Training model_classifer model....")
            print(model2file)
            train_e2e_model(model_classifer, model2file, inputs_train_x, inputs_train_y,
                            inputs_test_x, inputs_test_y,
                            Type_idex_word, test_target_count,
                            resultdir, npochos=100, batch_size=batch_size, retrain=False)



        print("test model_classifer model....")
        print(model2file)

        infer_e2e_model(model_classifer, model2name, model2file,
                        inputs_test_x, inputs_test_y, Type_idex_word,
                        test_target_count, resultdir, batch_size=32)




if __name__ == "__main__":

    maxlen = 50

    model1name = 'Model_BiLSTM_CRF'
    model2name = 'Model_LSTM_BiLSTM_LSTM'

    print(model1name)
    print(model2name)

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

    batch_size = 32
    hidden_dim = 200
    SecondTrain = True
    retrain = False
    Test = True

    Train41stsegment(datafname, datafile, trainfile, testfile, w2v_file, c2v_file, maxlen,
                     model1name, model1file,
                     resultdir, hidden_dim, batch_size, Test)







    # import tensorflow as tf
    # import keras.backend.tensorflow_backend as KTF
    #
    # KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))

    # CUDA_VISIBLE_DEVICES=1 python3 TrainModel.py

