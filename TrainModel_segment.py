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

from ProcessData_BIOES2F import get_data_4segment_BIOES
from Evaluate import evaluation_NER, evaluation_NER2, evaluation_NER_BIOES,evaluation_NER_Type
from network.NN_single import Model_BiLSTM_CRF, Model_BiLSTM_CnnDecoder, Model_BiLSTM_parallel_8_64_CRF
from network.NN_single import Model_BiLSTM_Softmax
from network.NN_single import Model_BiLSTM_X2_CRF



def test_model_segment(model, inputs_test_x, inputs_test_y, index2word, resultfile='', batch_size=256):

    index2word[0] = ''

    testresult2 = []
    predictions = model.predict(inputs_test_x, batch_size=batch_size)

    for si in range(0, len(predictions)):

        ptag_BIOES = []
        for word in predictions[si]:
            next_index = np.argmax(word)
            next_token = index2word[next_index]
            ptag_BIOES.append(next_token)

        ttag_BIOES = []
        for word in inputs_test_y[0][si]:
            next_index = np.argmax(word)
            next_token = index2word[next_index]
            ttag_BIOES.append(next_token)

        testresult2.append([ptag_BIOES, ttag_BIOES])


    P, R, F, PR_count, P_count, TR_count = evaluation_NER_BIOES(testresult2, resultfile=resultfile+'.BIORS.txt')
    print('divide---BIOES>>>>>>>>>>', P, R, F)

    return P, R, F, PR_count, P_count, TR_count


def train_e2e_model(model, modelfile, inputs, target_idex_word, resultdir, npochos=100, batch_size=50, retrain=False):

    inputs_train_x = inputs[0]
    inputs_train_y = inputs[1]
    inputs_test_x = inputs[2]
    inputs_test_y = inputs[3]

    if retrain:
        model.load_weights(modelfile)

    model.summary()

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

    while (epoch < npochos):
        epoch = epoch + save_inter

        history = model.fit(inputs_train_x, inputs_train_y,
                            batch_size=batch_size,
                            epochs=save_inter,
                            validation_split=0.2,
                            shuffle=True,
                            # sample_weight =sample_weight,
                            verbose=1)



        resultfile = ''

        print('the test result-----------------------')
        P, R, F, PR_count, P_count, TR_count = test_model_segment(model, inputs_test_x, inputs_test_y, target_idex_word, resultfile,
                                                      batch_size)

        if F > maxF:
            earlystopping = 0
            maxF=F
            model.save_weights(modelfile, overwrite=True)

        else:
            earlystopping += save_inter

        print(epoch, P, R, F, '  maxF=', maxF)

        if earlystopping >= 10:
            break

    return model


def infer_e2e_model(model, modelname, modelfile, inputs, target_idex_word, resultdir, batch_size=50):
    inputs_test_x = inputs[0]
    inputs_test_y = inputs[1]

    resultfile = resultdir + "result-" + modelname + '-' + str(datetime.datetime.now())+'.txt'

    model.load_weights(modelfile)

    P, R, F, PR_count, P_count, TR_count = test_model_segment(model, inputs_test_x, inputs_test_y, target_idex_word, resultfile,
                                                      batch_size)
    print('P= ', P, '  R= ', R, '  F= ', F)


def SelectModel(modelname, sourcevocabsize, targetvocabsize, source_W,
                             input_seq_lenth,
                             output_seq_lenth,
                             hidden_dim, emd_dim,
                     sourcecharsize,character_W,input_word_length,char_emd_dim, batch_size=32,
                     loss='categorical_crossentropy', optimizer='rmsprop'):
    nn_model = None

    if modelname is 'Model_BiLSTM_CRF':
        nn_model = Model_BiLSTM_CRF(sourcevocabsize=sourcevocabsize, targetvocabsize=targetvocabsize,
                                                       source_W=source_W,
                                                       input_seq_lenth=input_seq_lenth,
                                                       output_seq_lenth=output_seq_lenth,
                                                       hidden_dim=hidden_dim, emd_dim=emd_dim,
                                                        sourcecharsize=sourcecharsize,
                                                        character_W=character_W,
                                                        input_word_length=input_word_length,
                                                        char_emd_dim=char_emd_dim, batch_size=batch_size)

    elif modelname is 'Model_BiLSTM_X2_CRF':
        nn_model = Model_BiLSTM_X2_CRF(sourcevocabsize=sourcevocabsize, targetvocabsize=targetvocabsize,
                                                       source_W=source_W,
                                                       input_seq_lenth=input_seq_lenth,
                                                       output_seq_lenth=output_seq_lenth,
                                                       hidden_dim=hidden_dim, emd_dim=emd_dim,
                                                        sourcecharsize=sourcecharsize,
                                                        character_W=character_W,
                                                        input_word_length=input_word_length,
                                                        char_emd_dim=char_emd_dim, batch_size=batch_size)

    return nn_model


if __name__ == "__main__":

    maxlen = 50

    # modelname = 'Model_BiLSTM_CnnDecoder'
    modelname = 'Model_BiLSTM_CRF'
    # modelname = 'Model_BiLSTM_parallel_8_64_CRF'
    # modelname = 'Model_BiLSTM_Softmax'
    # modelname = 'Model_BiLSTM_X2_CRF'

    print(modelname)

    w2v_file = "./data/w2v/glove.6B.100d.txt"
    c2v_file = "./data/w2v/C0NLL2003.NER.c2v.txt"
    # trainfile = "./data/CoNLL2003_NER/eng.train.BIOES.txt"
    # devfile = "./data/CoNLL2003_NER/eng.testa.BIOES.txt"
    trainfile = "./data/CoNLL2003_NER/eng.train.testa.BIOES.txt"
    testfile = "./data/CoNLL2003_NER/eng.testb.BIOES.txt"
    resultdir = "./data/result/"

    datafile = "./model_data/data_segment_BIOES_PreC2V.1" + ".pkl"

    modelfile = "next ...."

    batch_size = 32
    retrain = False
    Test = True

    if not os.path.exists(datafile):
        print("Precess data....")
        get_data_4segment_BIOES(trainfile, testfile, w2v_file, c2v_file, datafile, w2v_k=100, c2v_k=50, maxlen=maxlen)

    for inum in range(0,3):

        modelfile = "./model/" + modelname + "__PreC2V" + "__segment_" + str(inum) + ".h5"

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

