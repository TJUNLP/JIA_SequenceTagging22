# -*- encoding:utf-8 -*-
import numpy as np
import pickle, os
import ProcessData_segment_PreC2V
import TrainModel_segment, TrainModel_tagging
from Evaluate import evaluation_NER, evaluation_NER2, evaluation_NER_BIOES,evaluation_NER_Type
import Seq2fragment
import ProcessData_S2F

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


def test_model_taggiing(model_2Step, testresult_1Step, testfile,
                        word_vob, word_idex_word, char_vob,
                        target_idex_word, Type_vob, index2type, max_context, max_fragment, hasNeg):


    test_fragment_list = Seq2fragment.Seq2frag4test(testresult_1Step, testfile, word_vob, target_vob, target_idex_word)
    print('len(test_fragment_list)---', len(test_fragment_list))

    test_2Step = ProcessData_S2F.make_idx_word_index(test_fragment_list, Type_vob, max_context, max_fragment, hasNeg=hasNeg)
    print(len(test_2Step))

    chartest_2Step = ProcessData_S2F.make_idx_char_index(test_fragment_list, max_context, max_fragment, max_c, char_vob, word_idex_word)
    print(len(chartest))

    testx_fragment = np.asarray(test_2Step[0], dtype="int32")
    testx_leftcontext = np.asarray(test_2Step[1], dtype="int32")
    testx_rightcontext = np.asarray(test_2Step[2], dtype="int32")
    testy = np.asarray(test_2Step[3], dtype="int32")
    testchar_fragment = np.asarray(chartest_2Step[0], dtype="int32")
    testchar_leftcontext = np.asarray(chartest_2Step[1], dtype="int32")
    testchar_rightcontext = np.asarray(chartest_2Step[2], dtype="int32")
    predictions = model_2Step.predict([testx_fragment, testx_leftcontext, testx_rightcontext,
                                   testchar_fragment, testchar_leftcontext, testchar_rightcontext],
                                      verbose=0)

    predict_right = 0
    predict = 0
    target = 5648

    for num, ptagindex in enumerate(predictions):

        next_index = np.argmax(ptagindex)
        ptag = index2type[next_index]

        if ptag != 'NULL':
            predict += 1

        if not any(testy[num]):
            ttag = 'NULL'

        else:
            ttag = index2type[np.argmax(testy[num])]

        if ptag == ttag  and ttag != 'NULL':
            predict_right += 1

    P = predict_right / predict
    R = predict_right / target
    F = 2 * P * R / (P + R)
    print('predict_right =, predict =, target =, len(predictions) =', predict_right, predict, target, len(predictions))
    print('P= ', P)
    print('R= ', R)
    print('F= ', F)


if __name__ == '__main__':

    testfile = "./data/CoNLL2003_NER/eng.testb.BIOES.txt"


    datafile_1Step = "./model_data/data_segment_BIOES_PreC2V.1" + ".pkl"

    modelname_1Step = 'Model_BiLSTM_CRF'
    inum = 0
    modelfile_1Step = "./model/" + modelname_1Step + "__PreC2V" + "__segment_" + str(inum) + ".h5"

    traindata, devdata, testdata, chartrain, chardev, chartest,\
    source_W, character_W,\
    source_vob, index2word, target_vob, index2tag, source_char,\
    max_s, w2v_k, max_c, c2v_k = pickle.load(open(datafile_1Step, 'rb'))

    batch_size_1Step =32

    model_1Step = TrainModel_segment.SelectModel(modelname_1Step,
                                             sourcevocabsize=len(source_vob),
                                             targetvocabsize=len(target_vob),
                          source_W=source_W,
                          input_seq_lenth=max_s,
                          output_seq_lenth=max_s,
                          hidden_dim=200, emd_dim=w2v_k,
                          sourcecharsize=len(source_char),
                          character_W=character_W,
                          input_word_length=max_c, char_emd_dim=c2v_k, batch_size=batch_size_1Step)

    model_1Step.load_weights(modelfile_1Step)

    testresult_1Step = test_model_segment(model_1Step, testdata, chartest, index2tag)

    hasNeg = True

    datafname = 'data_tagging_4type_PreC2V.1'
    if hasNeg:
        datafname = 'data_tagging_5type_PreC2V.PartErgodic.2' #'data_tagging_5type_PreC2V.1'
    datafile_2Step = "./model_data/" + datafname + ".pkl"
    modelname_2Step = 'Model_LSTM_BiLSTM_LSTM'
    inum = 0
    modelfile_2Step = "./model/" + modelname_2Step + "__" + datafname + "_tagging_" + str(inum) + ".h5"

    traindata, devdata, testdata,\
    chartrain, chardev, chartest,\
    word_vob, word_idex_word, \
    target_vob, target_idex_word, \
    Type_vob, Type_idex_word,\
    char_vob, char_idex_char,\
    word_W, word_k,\
    character_W, character_k,\
    max_context, max_fragment, max_c = pickle.load(open(datafile_2Step, 'rb'))

    batch_size_2Step =256 #32

    testx_fragment = np.asarray(testdata[0], dtype="int32")
    testx_leftcontext = np.asarray(testdata[1], dtype="int32")
    testx_rightcontext = np.asarray(testdata[2], dtype="int32")
    testy = np.asarray(testdata[3], dtype="int32")
    testchar_fragment = np.asarray(chartest[0], dtype="int32")
    testchar_leftcontext = np.asarray(chartest[1], dtype="int32")
    testchar_rightcontext = np.asarray(chartest[2], dtype="int32")

    model_2Step = TrainModel_tagging.SelectModel(modelname_2Step,
                          wordvocabsize=len(word_vob),
                          targetvocabsize=len(Type_vob),
                          charvobsize=len(char_vob),
                          word_W=word_W, char_W=character_W,
                          input_fragment_lenth=max_fragment,
                          input_leftcontext_lenth=max_context,
                          input_rightcontext_lenth=max_context,
                          input_maxword_length=max_c,
                          w2v_k=word_k, c2v_k=character_k,
                          hidden_dim=200, batch_size=batch_size_2Step)

    model_2Step.load_weights(modelfile_2Step)


    loss, acc = model_2Step.evaluate([testx_fragment, testx_leftcontext, testx_rightcontext,
                                      testchar_fragment, testchar_leftcontext, testchar_rightcontext],
                                     [testy],
                                     verbose=0,
                                     batch_size=10)

    print('\n test_test score:', loss, acc)

    test_model_taggiing(model_2Step, testresult_1Step, testfile,
                        word_vob, word_idex_word, char_vob,
                        target_idex_word, Type_vob, Type_idex_word, max_context, max_fragment, hasNeg=hasNeg)



