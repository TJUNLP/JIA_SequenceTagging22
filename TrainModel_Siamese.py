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
from PrecessData_Siamese import get_data
from Evaluate import evaluation_NER
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from network.NN_Siamese import Model_BiLSTM__MLP


def test_model(nn_model, fragment_test, target_vob, max_s, max_posi):

    predict = 0
    predict_right = 0
    totel_right = len(fragment_test)

    for frag in fragment_test:
        fragment_l = int(frag[0])
        fragment_r = int(frag[1])
        fragment_tag = target_vob[frag[2]]
        sent = frag[3]

        data_s = sent[0:min(len(sent), max_s)] + [0] * max(0, max_s - len(sent))

        list_left = [min(i, max_posi) for i in range(1, fragment_l+1)]
        list_left.reverse()
        feature_posi = list_left + [0 for i in range(fragment_l, fragment_r)] + \
                       [min(i, max_posi) for i in range(1, len(sent) - fragment_r + 1)]
        data_posi = feature_posi[0:min(len(sent), max_s)] + [max_posi] * max(0, max_s - len(sent))

        data_s_all = []
        data_posi_all = []
        data_tag_all = []
        for ins in target_vob.values():
            data_s_all.append(data_s)
            data_posi_all.append(data_posi)
            data_tag_all.append([ins])
        pairs = [data_s_all, data_posi_all, data_tag_all]

        x1_sent = np.asarray(pairs[0], dtype="int32")
        x1_posi = np.asarray(pairs[1], dtype="int32")
        x2_tag = np.asarray(pairs[2], dtype="int32")

        predictions = nn_model.predict([x1_sent, x1_posi, x2_tag], batch_size=4, verbose=0)

        mindis = 1
        mindis_where = 0
        for num, disvlaue in enumerate(predictions):
            if disvlaue < mindis:
                mindis = disvlaue
                mindis_where = pairs[2][num]
        if mindis < 0.5:
            predict += 1
            if mindis_where == fragment_tag:
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
                    inputs_dev_x, inputs_dev_y, fragment_test,
                    resultdir, type_vob, max_s, max_posi,
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

        print('the test result-----------------------')
        P, R, F = test_model(nn_model, fragment_test, type_vob, max_s, max_posi)

        if F > maxF:
            earlystop = 0
            maxF = F
            nn_model.save_weights(modelfile, overwrite=True)

        print(nowepoch, P, R, F, '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>maxF=', maxF)

        if earlystop >= 20:
            break

    return nn_model


def infer_e2e_model(nnmodel, modelname, modelfile, fragment_test, resultdir, type_vob, max_s, max_posi):

    nnmodel.load_weights(modelfile)
    resultfile = resultdir + "result-" + modelname + '-' + str(datetime.datetime.now())+'.txt'
    P, R, F = test_model(nnmodel, fragment_test, type_vob, max_s, max_posi)
    print('P = ', P, 'R = ', R, 'F = ', F)


def SelectModel(modelname, wordvocabsize, tagvocabsize, posivocabsize,
                     word_W, posi_W, tag_W,
                     input_sent_lenth,
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

    return nn_model


if __name__ == "__main__":

    maxlen = 50

    hasNeg = True

    modelname = 'Model_BiLSTM__MLP'

    print(modelname)

    w2v_file = "./data/w2v/glove.6B.100d.txt"
    c2v_file = "./data/w2v/C0NLL2003.NER.c2v.txt"
    trainfile = "./data/CoNLL2003_NER/eng.train.BIOES.txt"
    devfile = "./data/CoNLL2003_NER/eng.testa.BIOES.txt"
    testfile = "./data/CoNLL2003_NER/eng.testb.BIOES.txt"
    resultdir = "./data/result/"

    datafname = 'data_Siamese.1'
    datafile = "./model_data/" + datafname + ".pkl"

    modelfile = "next ...."

    batch_size = 512
    hidden_dim = 200
    SecondTrain = True
    retrain = False
    Test = True

    if not os.path.exists(datafile):
        print("Precess data....")

        get_data(trainfile, devfile, testfile, w2v_file, c2v_file, datafile,
                 w2v_k=100, c2v_k=50, maxlen=maxlen)

    pairs_train, labels_train, pairs_dev, labels_dev, fragment_test,\
    word_vob, word_id2word, word_W, w2v_k,\
    TYPE_vob, TYPE_id2type, type_W, type_k,\
    posi_W, posi_W,\
    target_vob, target_id2word, max_s, max_posi = pickle.load(open(datafile, 'rb'))

    train_x1_sent = np.asarray(pairs_train[0], dtype="int32")
    train_x1_posi = np.asarray(pairs_train[1], dtype="int32")
    train_x2_tag = np.asarray(pairs_train[2], dtype="int32")
    train_y = np.asarray(labels_train, dtype="int32")

    dev_x1_sent = np.asarray(pairs_dev[0], dtype="int32")
    dev_x1_posi = np.asarray(pairs_dev[1], dtype="int32")
    dev_x2_tag = np.asarray(pairs_dev[2], dtype="int32")
    dev_y = np.asarray(labels_dev, dtype="int32")

    nn_model = SelectModel(modelname,
                           wordvocabsize=len(word_vob),
                           tagvocabsize=len(TYPE_vob),
                           posivocabsize=max_posi+1,
                           word_W=word_W, posi_W=posi_W, tag_W=type_W,
                           input_sent_lenth=max_s,
                           w2v_k=w2v_k, posi2v_k=max_posi+1, tag2v_k=type_k,
                           batch_size=batch_size)

    inputs_train_x = [train_x1_sent, train_x1_posi, train_x2_tag]
    inputs_train_y = [train_y]

    inputs_dev_x = [dev_x1_sent, dev_x1_posi, dev_x2_tag]
    inputs_dev_y = [dev_y]


    for inum in range(0, 3):

        modelfile = "./model/" + modelname + "__" + datafname + "_tagging_" + str(inum) + ".h5"

        if not os.path.exists(modelfile):
            print("Lstm data has extisted: " + datafile)
            print("Training EE model....")
            print(modelfile)
            train_e2e_model(nn_model, modelfile, inputs_train_x, inputs_train_y,
                    inputs_dev_x, inputs_dev_y, fragment_test,
                    resultdir, TYPE_vob, max_s, max_posi, npoches=100, batch_size=50, retrain=False)

        else:
            if retrain:
                print("ReTraining EE model....")
                train_e2e_model(nn_model, modelfile, inputs_train_x, inputs_train_y,
                                inputs_dev_x, inputs_dev_y, fragment_test,
                                resultdir, TYPE_vob, max_s, max_posi, npoches=100, batch_size=50, retrain=False)

        if Test:
            print("test EE model....")
            print(datafile)
            print(modelfile)
            infer_e2e_model(nn_model, modelname, modelfile, fragment_test, resultdir)



    # import tensorflow as tf
    # import keras.backend.tensorflow_backend as KTF
    #
    # KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))

    # CUDA_VISIBLE_DEVICES=1 python3 TrainModel.py

