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
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from network.NN_tagging import Model_LSTM_BiLSTM_LSTM


def tagging4SecondTraining(nn_model, testdata, chardata, index2type):

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
                                      verbose=1)

    data_fragment_all = []
    data_leftcontext_all = []
    data_rightcontext_all = []
    data_t_all = []

    char_fragment_all = []
    char_leftcontext_all = []
    char_rightcontext_all = []

    for num, ptagindex in enumerate(predictions):

        next_index = np.argmax(ptagindex)
        ptag = index2type[next_index]


        ttag = index2type[np.argmax(testy[num])]

        if (ttag != 'NULL') or (ptag != 'NULL' and ttag == 'NULL'):
            data_fragment_all.append(testdata[0][num])
            data_leftcontext_all.append(testdata[1][num])
            data_rightcontext_all.append(testdata[2][num])
            data_t_all.append(testdata[3][num])
            char_fragment_all.append(chardata[0][num])
            char_leftcontext_all.append(chardata[1][num])
            char_rightcontext_all.append(chardata[2][num])

    SecondTrain_data = [data_fragment_all, data_leftcontext_all, data_rightcontext_all, data_t_all]
    SecondTrain_chardata = [char_fragment_all, char_leftcontext_all, char_rightcontext_all]

    print('count of 2nd data', len(char_fragment_all))

    return SecondTrain_data, SecondTrain_chardata



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
                    inputs_dev_x, inputs_dev_y, inputs_test_x, inputs_test_y,
                    devdata, chardev, Type_idex_word, dev_target_count,
                    testdata, chartest, test_target_count,
                    resultdir, npochos=100, batch_size=50, retrain=False):
    class_weight = {0: 50, 1: 50, 2: 50, 3: 50, 4: 1}
    if retrain:
        nn_model.load_weights(modelfile)
        modelfile = modelfile + '.2nd.h5'
        class_weight = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}

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
    epochlen = 1
    while (epoch < npochos):
        epoch = epoch + epochlen
        i += 1
        # if os.path.exists(modelfile):
        #     nn_model.load_weights(modelfile)

        checkpointer = ModelCheckpoint(filepath=modelfile + ".best_model.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True)
        history = nn_model.fit(inputs_train_x,
                               inputs_train_y,
                               batch_size=batch_size,
                               epochs=epochlen,
                               validation_data=(inputs_dev_x, inputs_dev_y),
                               shuffle=True,
                               class_weight=None,
                               verbose=1,
                               # callbacks=[checkpointer]
                               )

        print('the dev result-----------------------')
        P_dev, R_dev, F_dev = test_model_tagging(nn_model, devdata, chardev, Type_idex_word, dev_target_count)

        print('the test result-----------------------')
        loss, acc = nn_model.evaluate(inputs_test_x, inputs_test_y, verbose=1, batch_size=512)
        print('\n test_test score:', loss, acc)
        P, R, F = test_model_tagging(nn_model, testdata, chartest, Type_idex_word, test_target_count)

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

        # if P_dev >= 0.45 and R_dev >= 0.95:
        #     nn_model.save_weights(modelfile, overwrite=True)
        #     break

    return nn_model


def infer_e2e_model(nnmodel, modelfile,
                    inputs_dev_x, inputs_dev_y, inputs_test_x, inputs_test_y,
                    devdata, chardev, Type_idex_word, dev_target_count,
                    testdata, chartest, test_target_count,
                    resultdir, batch_size=32):


    nnmodel.load_weights(modelfile)
    # nnmodel = load_model(lstm_modelfile)

    resultfile = resultdir + "result-" + modelname + '-' + str(datetime.datetime.now())+'.txt'

    loss, acc = nnmodel.evaluate(inputs_dev_x, inputs_dev_y, verbose=0,
                                  batch_size=512)
    print('\n test_dev score:', loss, acc)
    test_model_tagging(nnmodel, devdata, chardev, Type_idex_word, dev_target_count)

    loss, acc = nnmodel.evaluate(inputs_test_x, inputs_test_y, verbose=0,
                                  batch_size=512)
    print('\n test_test score:', loss, acc)
    test_model_tagging(nnmodel, testdata, chartest, Type_idex_word, test_target_count)


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

    hasNeg = False

    modelname = 'Model_LSTM_BiLSTM_LSTM'

    print(modelname)

    w2v_file = "./data/w2v/glove.6B.100d.txt"
    c2v_file = "./data/w2v/C0NLL2003.NER.c2v.txt"
    trainfile = "./data/CoNLL2003_NER/eng.train.BIOES.txt"
    devfile = "./data/CoNLL2003_NER/eng.testa.BIOES.txt"
    testfile = "./data/CoNLL2003_NER/eng.testb.BIOES.txt"
    resultdir = "./data/result/"

    datafname = 'data_tagging_4type_PreC2V.1'

    if hasNeg:
        datafname = 'data_tagging_5type_PreC2V.Ergodic.1'
    datafile = "./model_data/" + datafname + ".pkl"

    modelfile = "next ...."

    batch_size = 512
    hidden_dim = 200
    SecondTrain = False
    retrain = False
    Test = True

    if not os.path.exists(datafile):
        print("Precess data....")

        get_data(trainfile,devfile, testfile, w2v_file, c2v_file, datafile,
                 w2v_k=100, c2v_k=50, maxlen=maxlen, hasNeg=hasNeg)

    traindata, devdata, testdata,\
    chartrain, chardev, chartest,\
    word_vob, word_idex_word, \
    target_vob, target_idex_word, \
    Type_vob, Type_idex_word,\
    char_vob, char_idex_char,\
    word_W, word_k,\
    character_W, character_k,\
    max_context, max_fragment, max_c, \
    train_target_count, dev_target_count, test_target_count = pickle.load(open(datafile, 'rb'))

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

    inputs_train_x = [trainx_fragment, trainx_leftcontext, trainx_rightcontext,
                    trainchar_fragment, trainchar_leftcontext, trainchar_rightcontext]
    inputs_train_y = [trainy]

    inputs_dev_x = [devx_fragment, devx_leftcontext, devx_rightcontext,
                    devchar_fragment, devchar_leftcontext, devchar_rightcontext]
    inputs_dev_y = [devy]

    inputs_test_x = [testx_fragment, testx_leftcontext, testx_rightcontext,
                     testchar_fragment, testchar_leftcontext, testchar_rightcontext]
    inputs_test_y = [testy]



    for inum in range(9, 12):

        modelfile = "./model/" + modelname + "__" + datafname + "_tagging_" + str(inum) + ".h5"

        if not os.path.exists(modelfile):
            print("Lstm data has extisted: " + datafile)
            print("Training EE model....")
            print(modelfile)
            train_e2e_model(nn_model, modelfile, inputs_train_x, inputs_train_y,
                            inputs_dev_x, inputs_dev_y, inputs_test_x, inputs_test_y,
                            devdata, chardev, Type_idex_word, dev_target_count,
                            testdata, chartest, test_target_count,
                            resultdir, npochos=100, batch_size=batch_size, retrain=False)

        else:
            if retrain:
                print("ReTraining EE model....")
                train_e2e_model(nn_model, modelfile, inputs_train_x, inputs_train_y,
                            inputs_dev_x, inputs_dev_y, inputs_test_x, inputs_test_y,
                            devdata, chardev, Type_idex_word, dev_target_count,
                            testdata, chartest, test_target_count,
                            resultdir, npochos=100, batch_size=batch_size, retrain=False)

        if Test:
            print("test EE model....")
            print(datafile)
            print(modelfile)
            infer_e2e_model(nn_model, modelfile,
                            inputs_dev_x, inputs_dev_y, inputs_test_x, inputs_test_y,
                            devdata, chardev, Type_idex_word, dev_target_count,
                            testdata, chartest, test_target_count, resultdir, batch_size=batch_size)

        if SecondTrain:

            Train_2ndT_data, Train_2ndT_chardata = tagging4SecondTraining(nn_model, traindata, chartrain, Type_idex_word)
            Dev_2ndT_data, Dev_2ndT_chardata = tagging4SecondTraining(nn_model, devdata, chardev, Type_idex_word)
            Test_2ndT_data, Test_2ndT_chardata = tagging4SecondTraining(nn_model, testdata, chartest, Type_idex_word)

            train2ndx_fragment = np.asarray(Train_2ndT_data[0], dtype="int32")
            train2ndx_leftcontext = np.asarray(Train_2ndT_data[1], dtype="int32")
            train2ndx_rightcontext = np.asarray(Train_2ndT_data[2], dtype="int32")
            train2ndy = np.asarray(Train_2ndT_data[3], dtype="int32")
            train2ndchar_fragment = np.asarray(Train_2ndT_chardata[0], dtype="int32")
            train2ndchar_leftcontext = np.asarray(Train_2ndT_chardata[1], dtype="int32")
            train2ndchar_rightcontext = np.asarray(Train_2ndT_chardata[2], dtype="int32")

            dev2ndx_fragment = np.asarray(Dev_2ndT_data[0], dtype="int32")
            dev2ndx_leftcontext = np.asarray(Dev_2ndT_data[1], dtype="int32")
            dev2ndx_rightcontext = np.asarray(Dev_2ndT_data[2], dtype="int32")
            dev2ndy = np.asarray(Dev_2ndT_data[3], dtype="int32")
            dev2ndchar_fragment = np.asarray(Dev_2ndT_chardata[0], dtype="int32")
            dev2ndchar_leftcontext = np.asarray(Dev_2ndT_chardata[1], dtype="int32")
            dev2ndchar_rightcontext = np.asarray(Dev_2ndT_chardata[2], dtype="int32")

            test2ndx_fragment = np.asarray(Test_2ndT_data[0], dtype="int32")
            test2ndx_leftcontext = np.asarray(Test_2ndT_data[1], dtype="int32")
            test2ndx_rightcontext = np.asarray(Test_2ndT_data[2], dtype="int32")
            test2ndy = np.asarray(Test_2ndT_data[3], dtype="int32")
            test2ndchar_fragment = np.asarray(Test_2ndT_chardata[0], dtype="int32")
            test2ndchar_leftcontext = np.asarray(Test_2ndT_chardata[1], dtype="int32")
            test2ndchar_rightcontext = np.asarray(Test_2ndT_chardata[2], dtype="int32")

            inputs_train2nd_x = [train2ndx_fragment, train2ndx_leftcontext, train2ndx_rightcontext,
                              train2ndchar_fragment, train2ndchar_leftcontext, train2ndchar_rightcontext]
            inputs_train2nd_y = [train2ndy]

            inputs_dev2nd_x = [dev2ndx_fragment, dev2ndx_leftcontext, dev2ndx_rightcontext,
                            dev2ndchar_fragment, dev2ndchar_leftcontext, dev2ndchar_rightcontext]
            inputs_dev2nd_y = [dev2ndy]

            inputs_test2nd_x = [test2ndx_fragment, test2ndx_leftcontext, test2ndx_rightcontext,
                             test2ndchar_fragment, test2ndchar_leftcontext, test2ndchar_rightcontext]
            inputs_test2nd_y = [test2ndy]

            if not os.path.exists(modelfile + '.2nd.h5'):
                print("2nd Training EE model....")

                train_e2e_model(nn_model, modelfile, inputs_train2nd_x, inputs_train2nd_y,
                                inputs_dev2nd_x, inputs_dev2nd_y, inputs_test2nd_x, inputs_test2nd_y,
                                Dev_2ndT_data, Dev_2ndT_chardata, Type_idex_word, dev_target_count,
                                Test_2ndT_data, Test_2ndT_chardata, test_target_count,
                                resultdir, npochos=100, batch_size=64, retrain=SecondTrain)

            if Test:
                print("2nd test EE model....")
                print(datafile)
                print(modelfile+ '.2nd.h5')
                infer_e2e_model(nn_model, modelfile+ '.2nd.h5',
                                inputs_dev2nd_x, inputs_dev2nd_y, inputs_test2nd_x, inputs_test2nd_y,
                                Dev_2ndT_data, Dev_2ndT_chardata, Type_idex_word, dev_target_count,
                                Test_2ndT_data, Test_2ndT_chardata, test_target_count, resultdir, batch_size=batch_size)




    # import tensorflow as tf
    # import keras.backend.tensorflow_backend as KTF
    #
    # KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))

    # CUDA_VISIBLE_DEVICES=1 python3 TrainModel.py

