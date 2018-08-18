# -*- encoding:utf-8 -*-

import pickle
import os.path
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from PrecessData import get_data, make_idx_data_index
from Evaluate import evaluation_NER, evaluation_NER2
# from keras.models import Sequential
# from keras.layers.embeddings import Embedding
from keras.layers import Flatten,Lambda,Conv2D
from keras.layers.core import Dropout, Activation, Permute, RepeatVector
from keras.legacy.layers import Merge
from keras.layers.merge import concatenate, Concatenate, multiply, Dot
from keras.layers import TimeDistributed, Input, Bidirectional, Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D, RepeatVector, AveragePooling1D
from keras.models import Model
from keras_contrib.layers import CRF
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
from keras import regularizers
# from keras.losses import my_cross_entropy_withWeight

def get_training_batch_xy_bias(inputsX, entlabel_train, inputsY, max_s, max_t,
                               batchsize, vocabsize, target_idex_word, lossnum, shuffle=False):
    assert len(inputsX) == len(inputsY)
    indices = np.arange(len(inputsX))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputsX) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        x_word = np.zeros((batchsize, max_s)).astype('int32')
        x_entl = np.zeros((batchsize, max_s)).astype('int32')
        y = np.zeros((batchsize, max_t, vocabsize + 1)).astype('int32')
        y = np.zeros((batchsize, max_t, vocabsize)).astype('int32')

        for idx, s in enumerate(excerpt):
            x_word[idx,] = inputsX[s]
            x_entl[idx,] = entlabel_train[s]
            for idx2, word in enumerate(inputsY[s]):
                targetvec = np.zeros(vocabsize + 1)
                targetvec = np.zeros(vocabsize)

                wordstr = ''

                if word != 0:
                    wordstr = target_idex_word[word]
                if wordstr.__contains__("E"):
                    targetvec[word] = lossnum
                else:
                    targetvec[word] = 1
                y[idx, idx2,] = targetvec
        if x_word is None:
            print("x is none !!!!!!!!!!!!!!")
        yield x_word, x_entl, y


def my_cross_entropy_Weight(y_true, y_pred, e=2):
    # 10--0.9051120758394614
    # 5--0.9068666845254041
    # 3--0.9066785396260019
    #0.5--0.9089935760171306
    # 0.3--0.9072770998485793
    # 0.7--0.9078959054978238
    # 0.6--0.9069292875521249
    next_index = np.argmax(y_true)
    # while 0<1:
    #     print('123')
    print('next_index-', next_index)
    if next_index == 0 or next_index == 1:
        return e * K.categorical_crossentropy(y_true, y_pred)
    else:
        return K.categorical_crossentropy(y_true, y_pred)


def get_training_xy_otherset(i, inputsX, inputsY,
                             inputsX_O, inputsY_O,
                             max_s, max_c, chartrain, chardev, pos_train, pos_dev, vocabsize, target_idex_word,sample_weight_value=1, shuffle=False):

    # AllX = np.concatenate((inputsX, inputsX_O), axis=0)
    # AllY = np.concatenate((inputsY, inputsY_O), axis=0)
    # AllChar = np.concatenate((chartrain, chardev), axis=0)
    # print('AllX.shape', AllX.shape, 'AllY.shape', AllY.shape)
    #
    # separate = int(AllX.__len__() / 9)
    # start = 0 + separate * (i % 9)
    # end = start + separate
    # print(start, end)
    #
    # inputsX = np.concatenate((AllX[:start], AllX[end:]), axis=0)
    # inputsX_O = AllX[start:end]
    # inputsY = np.concatenate((AllY[:start], AllY[end:]), axis=0)
    # inputsY_O = AllY[start:end]
    # chartrain = np.concatenate((AllChar[:start], AllChar[end:]), axis=0)
    # chardev = AllChar[start:end]




    # get any other set as validtest set
    assert len(inputsX) == len(inputsY)
    indices = np.arange(len(inputsX))
    if shuffle:
        np.random.shuffle(indices)


    x_train = np.zeros((len(inputsX), max_s)).astype('int32')
    # x_entl_train = np.zeros((len(inputsX), max_s)).astype('int32')
    x_pos_train = np.zeros((len(inputsX), max_s, 3)).astype('int32')
    y_train = np.zeros((len(inputsX), max_s, vocabsize + 1)).astype('int32')
    input_char = np.zeros((len(inputsX), max_s, max_c)).astype('int32')
    # print(inputsX.__len__())
    sample_weight = np.zeros((len(inputsX), max_s)).astype('int32')

    for idx, s in enumerate(indices):
        x_train[idx,] = inputsX[s]
        # print(idx,s)
        input_char[idx,] = chartrain[s]

        # x_entl_train[idx,] = entlabel_train[s]
        x_pos_train[idx,] = pos_train[s]

        for idx2, word in enumerate(inputsY[s]):
            targetvec = np.zeros(vocabsize + 1)
            # targetvec = np.zeros(vocabsize)

            if word != 0:
                wordstr = ''
                wordstr = target_idex_word[word]

                if wordstr.__contains__("O"):
                    targetvec[word] = 1
                    sample_weight[idx, idx2] = 1
                else:
                    targetvec[word] = 1
                    sample_weight[idx, idx2] = sample_weight_value
            else:
                targetvec[word] = 1
                sample_weight[idx,idx2] = 1

            # print('targetvec',targetvec)
            y_train[idx, idx2,] = targetvec


    x_word = x_train[:]
    y = y_train[:]
    # x_entl = x_entl_train[:]
    # x_pos = x_pos_train[:]

    assert len(inputsX_O) == len(inputsY_O)
    indices_O = np.arange(len(inputsX_O))
    x_train_O = np.zeros((len(inputsX_O), max_s)).astype('int32')
    input_char_O = np.zeros((len(inputsX_O), max_s, max_c)).astype('int32')
    # x_entl_train_O = np.zeros((len(inputsX_O), max_s)).astype('int32')
    x_pos_dev = np.zeros((len(inputsX_O), max_s, 3)).astype('int32')
    y_train_O = np.zeros((len(inputsX_O), max_s, vocabsize + 1)).astype('int32')

    for idx, s in enumerate(indices_O):
        x_train_O[idx,] = inputsX_O[s]
        input_char_O[idx,] = chardev[s]
        # x_entl_train_O[idx,] = entlabel_train_O[s]
        x_pos_dev[idx,] = pos_dev[s]
        for idx2, word in enumerate(inputsY_O[s]):
            targetvec = np.zeros(vocabsize + 1)

            if word != 0:
                wordstr = ''
                wordstr = target_idex_word[word]

                if wordstr.__contains__("O"):
                    targetvec[word] = 1
                else:
                    targetvec[word] = 1
            else:
                targetvec[word] = 1

            # print('targetvec',targetvec)
            y_train_O[idx, idx2,] = targetvec

    x_word_val = x_train_O[:]
    y_val = y_train_O[:]
    # x_entl_val = x_entl_train_O[:]
    # x_posl_val = x_posl_train_O[:]

    yield x_word, y, x_word_val , y_val, input_char, input_char_O, x_pos_train, x_pos_dev,sample_weight
    # return x_word, y , x_word_val , y_val


def get_training_xy(inputsX, poslabel_train, entlabel_train, inputsY, max_s, max_t, vocabsize, target_idex_word,
                    shuffle=False):
    # get 0.2 of trainset as validtest set
    assert len(inputsX) == len(inputsY)
    indices = np.arange(len(inputsX))
    if shuffle:
        np.random.shuffle(indices)

    inputsX = inputsX[indices]
    inputsY = inputsY[indices]
    entlabel_train = entlabel_train[indices]
    poslabel_train = poslabel_train[indices]

    x_train = np.zeros((len(inputsX), max_s)).astype('int32')
    x_entl_train = np.zeros((len(inputsX), max_s)).astype('int32')
    x_posl_train = np.zeros((len(inputsX), max_s)).astype('int32')
    y_train = np.zeros((len(inputsX), max_t, vocabsize + 1)).astype('int32')

    for idx, s in enumerate(indices):
        x_train[idx,] = inputsX[s]
        x_entl_train[idx,] = entlabel_train[s]
        x_posl_train[idx,] = poslabel_train[s]
        for idx2, word in enumerate(inputsY[s]):
            targetvec = np.zeros(vocabsize + 1)
            # targetvec = np.zeros(vocabsize)

            if word != 0:
                wordstr = ''
                wordstr = target_idex_word[word]

                if wordstr.__contains__("O"):
                    targetvec[word] = 1
                else:
                    targetvec[word] = 1
            else:
                targetvec[word] = 1

            # print('targetvec',targetvec)
            y_train[idx, idx2,] = targetvec

    num_validation_samples = int(0.2 * len(inputsX))
    x_word = x_train[:-num_validation_samples]
    y = y_train[:-num_validation_samples]
    x_entl = x_entl_train[:-num_validation_samples]
    x_posl = x_posl_train[:-num_validation_samples]

    x_word_val = x_train[-num_validation_samples:]
    y_val = y_train[-num_validation_samples:]
    x_entl_val = x_entl_train[-num_validation_samples:]
    x_posl_val = x_posl_train[-num_validation_samples:]

    return x_word, x_posl, x_entl, y, x_word_val, x_posl_val, x_entl_val, y_val


def creat_Model_BiLSTM_CnnDecoder(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim,
                              sourcecharsize, character_W, input_word_length, char_emd_dim,
                              sourcepossize, pos_W, pos_emd_dim,
                              loss='categorical_crossentropy', optimizer='rmsprop'):

    # 0.8349149507609669--attention,lstm*2decoder

    # pos_input = Input(shape=(input_seq_lenth,), dtype='int32')
    # pos_embeding = Embedding(input_dim=sourcepossize + 1,
    #                               output_dim=pos_emd_dim,
    #                               input_length=input_seq_lenth,
    #                               mask_zero=False,
    #                               trainable=True,
    #                               weights=[pos_W])(pos_input)

    word_input = Input(shape=(input_seq_lenth,), dtype='int32')

    char_input = Input(shape=(input_seq_lenth, input_word_length,), dtype='int32')

    char_embedding = Embedding(input_dim=sourcecharsize, output_dim=char_emd_dim,
                               batch_input_shape=(batch_size, input_seq_lenth, input_word_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[character_W])

    char_embedding2 = TimeDistributed(char_embedding)(char_input)

    char_cnn = TimeDistributed(Conv1D(50, 3, activation='relu', border_mode='valid'))(char_embedding2)

    char_macpool = TimeDistributed(GlobalMaxPooling1D())(char_cnn)
    # char_macpool = Dropout(0.5)(char_macpool)

    pos_input = Input(shape=(input_seq_lenth, 3,), dtype='int32')
    pos_embedding = Embedding(input_dim=sourcepossize+ 1,
                             output_dim=pos_emd_dim,
                               batch_input_shape=(batch_size, input_seq_lenth, 3),
                               mask_zero=False,
                               trainable=True,
                             weights=[pos_W])
    pos_embedding2 = TimeDistributed(pos_embedding)(pos_input)
    pos_cnn = TimeDistributed(Conv1D(20, 2, activation='relu', border_mode='valid'))(pos_embedding2)
    pos_macpool = TimeDistributed(GlobalMaxPooling1D())(pos_cnn)

    word_embedding_RNN = Embedding(input_dim=sourcevocabsize + 1, output_dim=emd_dim, input_length=input_seq_lenth,
                                   mask_zero=False, trainable=False, weights=[source_W])(word_input)
    # word_embedding_RNN = Dropout(0.5)(word_embedding_RNN)

    embedding = concatenate([word_embedding_RNN, char_macpool, pos_macpool], axis=-1)
    embedding = Dropout(0.5)(embedding)

    BiLSTM = Bidirectional(LSTM(int(hidden_dim / 2), return_sequences=True), merge_mode='concat')(embedding)
    BiLSTM = BatchNormalization()(BiLSTM)
    # BiLSTM = Dropout(0.3)(BiLSTM)

    # decodelayer1 = LSTM(50, return_sequences=False, go_backwards=True)(concat_LC_d)#!!!!!
    # repeat_decodelayer1 = RepeatVector(input_seq_lenth)(decodelayer1)
    # concat_decoder = concatenate([concat_LC_d, repeat_decodelayer1], axis=-1)#!!!!
    # decodelayer2 = LSTM(hidden_dim, return_sequences=True)(concat_decoder)
    # decodelayer = Dropout(0.5)(decodelayer2)

    # decoderlayer1 = LSTM(50, return_sequences=True, go_backwards=False)(BiLSTM)
    decoderlayer5 = Conv1D(50, 5, activation='relu', strides=1, padding='same')(BiLSTM)
    decoderlayer2 = Conv1D(50, 2, activation='relu', strides=1, padding='same')(BiLSTM)
    decoderlayer3 = Conv1D(50, 3, activation='relu', strides=1, padding='same')(BiLSTM)
    decoderlayer4 = Conv1D(50, 4, activation='relu', strides=1, padding='same')(BiLSTM)
    # 0.8868111121100423
    decodelayer = concatenate([decoderlayer2, decoderlayer3, decoderlayer4, decoderlayer5], axis=-1)
    decodelayer = BatchNormalization()(decodelayer)
    decodelayer = Dropout(0.5)(decodelayer)

    TimeD = TimeDistributed(Dense(targetvocabsize + 1))(decodelayer)
    # TimeD = Dropout(0.5)(TimeD)
    model = Activation('softmax')(TimeD)  # 0.8769744561783556

    # crf = CRF(targetvocabsize + 1, sparse_target=False)
    # model = crf(TimeD)

    Models = Model([word_input, char_input, pos_input], model)

    # Models.compile(loss=my_cross_entropy_Weight, optimizer='adam', metrics=['acc'])
    Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    # Models.compile(loss=loss, optimizer='adam', metrics=['acc'], sample_weight_mode="temporal")
    # Models.compile(loss=loss, optimizer=optimizers.RMSprop(lr=0.01), metrics=['acc'])
    # Models.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])
    # Models.compile(loss=crf.loss_function, optimizer=optimizers.RMSprop(lr=0.005), metrics=[crf.accuracy])

    return Models


def Model_BiLSTM_CnnDecoder_multi2(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim,
                              sourcecharsize, character_W, input_word_length, char_emd_dim,
                              sourcepossize, pos_W, pos_emd_dim,
                              loss='categorical_crossentropy', optimizer='rmsprop'):


    word_input = Input(shape=(input_seq_lenth,), dtype='int32')
    char_input = Input(shape=(input_seq_lenth, input_word_length,), dtype='int32')

    char_embedding = Embedding(input_dim=sourcecharsize, output_dim=char_emd_dim,
                               batch_input_shape=(batch_size, input_seq_lenth, input_word_length),
                               mask_zero=True,
                               trainable=True,
                               weights=[character_W])
    char_embedding2 = TimeDistributed(char_embedding)(char_input)
    char_cnn = TimeDistributed(Conv1D(50, 3, activation='relu', border_mode='valid'))(char_embedding2)
    char_macpool = TimeDistributed(GlobalMaxPooling1D())(char_cnn)
    # char_macpool = Dropout(0.5)(char_macpool)

    pos_input = Input(shape=(input_seq_lenth, 3,), dtype='int32')
    pos_embedding = Embedding(input_dim=sourcepossize+ 1,
                             output_dim=pos_emd_dim,
                               batch_input_shape=(batch_size, input_seq_lenth, 3),
                               mask_zero=True,
                               trainable=True,
                             weights=[pos_W])
    pos_embedding2 = TimeDistributed(pos_embedding)(pos_input)
    pos_cnn = TimeDistributed(Conv1D(20, 2, activation='relu', border_mode='valid'))(pos_embedding2)
    pos_macpool = TimeDistributed(GlobalMaxPooling1D())(pos_cnn)

    word_embedding_RNN = Embedding(input_dim=sourcevocabsize + 1, output_dim=emd_dim, input_length=input_seq_lenth,
                                   mask_zero=False, trainable=False, weights=[source_W])(word_input)
    # word_embedding_RNN = Dropout(0.5)(word_embedding_RNN)

    embedding = concatenate([word_embedding_RNN, char_macpool], axis=-1)
    embedding_dropout = Dropout(0.5)(embedding)

    LSTM_l = LSTM(int(hidden_dim / 2), return_sequences=True,go_backwards=False)(embedding_dropout)
    LSTM_r = LSTM(int(hidden_dim / 2), return_sequences=True, go_backwards=True)(embedding_dropout)
    # BiLSTM = BatchNormalization()(BiLSTM)

    encoder_concat = concatenate([LSTM_l, embedding, LSTM_r], axis=-1)
    encoder = Dropout(0.5)(encoder_concat)
    # encoder = TimeDistributed(Dense(100, activation='tanh'))(encoder)

    decoderlayer1 = Conv1D(50, 1, activation='relu', strides=1, padding='same')(encoder)
    decoderlayer2 = Conv1D(50, 2, activation='relu', strides=1, padding='same')(encoder)
    decoderlayer3 = Conv1D(50, 3, activation='relu', strides=1, padding='same')(encoder)
    decoderlayer4 = Conv1D(50, 4, activation='relu', strides=1, padding='same')(encoder)

    decodelayer = concatenate([decoderlayer1, decoderlayer2, decoderlayer3, decoderlayer4], axis=-1)
    decodelayer = BatchNormalization()(decodelayer)
    decodelayer = Dropout(0.5)(decodelayer)

    '''
    # decoder = TimeDistributed(Dense(100, activation='tanh'))(decodelayer)
    output = TimeDistributed(Dense(targetvocabsize + 1, activation='softmax'))(decodelayer)
    Models = Model([word_input, char_input], output)
    # Models.compile(loss=my_cross_entropy_Weight, optimizer='adam', metrics=['acc'])
    Models.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    # Models.compile(loss=loss, optimizer='adam', metrics=['acc'], sample_weight_mode="temporal")
    # Models.compile(loss=loss, optimizer=optimizers.RMSprop(lr=0.01), metrics=['acc'])
    # Models.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])
    # Models.compile(loss=crf.loss_function, optimizer=optimizers.RMSprop(lr=0.005), metrics=[crf.accuracy])

    '''
    '''
    mlp2_hidden1 = TimeDistributed(Dense(50, activation='tanh'))(decodelayer)
    output1 = TimeDistributed(Dense(2 + 1, activation='sigmoid'), name='isOther')(decodelayer)

    mlp1_hidden1 = TimeDistributed(Dense(100, activation='relu'))(decodelayer)
    mlp1_concat = concatenate([mlp2_hidden1, mlp1_hidden1], axis=-1)
    output12 = TimeDistributed(Dense(targetvocabsize + 1, activation='softmax'), name='finall')(decodelayer)
    
    Models = Model([word_input, char_input], [output12, output1])
    Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
                   loss={'finall': 'categorical_crossentropy', 'isOther': 'binary_crossentropy'},
                   loss_weights={'finall': 1., 'isOther': 1.},
                   metrics={'finall': ['acc'], 'isOther': ['acc']})
    '''
    '''
    mlp2_hidden1 = TimeDistributed(Dense(100, activation='tanh'))(encoder_concat)
    output1 = TimeDistributed(Dense(targetvocabsize + 1, activation='softmax'), name='OP1')(decodelayer)

    BiLSTM = Bidirectional(LSTM(100, return_sequences=True), merge_mode='concat')(output1)
    BiLSTM = Dropout(0.5)(BiLSTM)

    output1_label = Lambda(lambda xin: K.argmax(xin, axis=-1))(output1)
    output1_label = Lambda(lambda xin: K.cast(xin, dtype='float32'))(output1_label)
    # output1_label = Dense(50, activation='tanh')(output1_label)
    output1_label_rep = RepeatVector(input_seq_lenth)(output1_label)

    # mlp1_hidden1 = TimeDistributed(Dense(100, activation='tanh'))(encoder_concat)
    mlp1_concat = concatenate([BiLSTM, decodelayer], axis=-1)
    output12 = TimeDistributed(Dense(targetvocabsize + 1, activation='softmax'), name='OP2')(mlp1_concat)

    Models = Model([word_input, char_input], [output12, output1])
    Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
                   loss={'OP2': 'categorical_crossentropy', 'OP1': 'categorical_crossentropy'},
                   loss_weights={'OP2': 1., 'OP1': 1.},
                   metrics={'OP2': ['acc'], 'OP1': ['acc']})
    '''
    mlp2_hidden1 = TimeDistributed(Dense(50, activation='tanh'))(decodelayer)
    output1 = TimeDistributed(Dense(2 + 1, activation='sigmoid'), name='isOther')(decodelayer)

    output2 = TimeDistributed(Dense(5 + 1))(decodelayer)
    output2 = Activation('softmax', name='BIOES')(output2)

    mlp1_hidden1 = TimeDistributed(Dense(100, activation='relu'))(decodelayer)
    mlp1_concat = concatenate([mlp2_hidden1, mlp1_hidden1], axis=-1)
    output12 = TimeDistributed(Dense(targetvocabsize + 1, activation='softmax'), name='finall')(decodelayer)

    Models = Model([word_input, char_input], [output12, output1, output2])
    Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
                   loss={'finall': 'categorical_crossentropy', 'isOther': 'binary_crossentropy', 'BIOES': 'categorical_crossentropy'},
                   loss_weights={'finall': 1., 'isOther': 1., 'BIOES': 1.},
                   metrics={'finall': ['acc'], 'isOther': ['acc'], 'BIOES': ['acc']})

    return Models



def creat_Model_BiLSTM_CNN_concat(sourcevocabsize, targetvocabsize, poslabelvobsize, entlabelvobsize, source_W, poslabel_W,
                              entlabel_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim, loss='categorical_crossentropy', optimizer='rmsprop'):
    # BiLSTM(w,p)+CNN(w,p)--timestep of LSTM concat CNN-GlobalMaxPool--softmax
    word_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posl_input = Input(shape=(input_seq_lenth,), dtype='int32')
    entl_input = Input(shape=(input_seq_lenth,), dtype='int32')

    l_A_embedding = Embedding(input_dim=sourcevocabsize + 1, output_dim=emd_dim, input_length=input_seq_lenth,
                              mask_zero=True, trainable=True, weights=[source_W])(word_input)
    l_A_embedding_CNN = Embedding(input_dim=sourcevocabsize + 1, output_dim=emd_dim, input_length=input_seq_lenth,
                              mask_zero=False, trainable=True, weights=[source_W])(word_input)

    poslable_embeding = Embedding(input_dim=poslabelvobsize + 1, output_dim=poslabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=True, trainable=True, weights=[poslabel_W])(posl_input)
    poslable_embeding_CNN = Embedding(input_dim=poslabelvobsize + 1, output_dim=poslabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=False, trainable=True, weights=[poslabel_W])(posl_input)

    entlable_embeding = Embedding(input_dim=entlabelvobsize + 1, output_dim=entlabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=True, trainable=True, weights=[entlabel_W])(entl_input)
    entlable_embeding_CNN = Embedding(input_dim=entlabelvobsize + 1, output_dim=entlabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=False, trainable=True, weights=[entlabel_W])(entl_input)

    concat_input = concatenate([l_A_embedding, poslable_embeding, entlable_embeding], axis=-1)
    concat_input = Dropout(0.3)(concat_input)
    concat_input_CNN = concatenate([l_A_embedding_CNN, poslable_embeding_CNN, entlable_embeding_CNN], axis=-1)
    concat_input_CNN = Dropout(0.3)(concat_input_CNN)

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True, dropout=0.1))(concat_input)

    cnn = Conv1D(hidden_dim, 3, activation='relu', strides=1, padding='same')(concat_input_CNN)
    maxpool =  GlobalMaxPooling1D()(cnn)
    repeat_maxpool = RepeatVector(input_seq_lenth)(maxpool)

    concat_LC = concatenate([BiLSTM, repeat_maxpool], axis=-1)
    concat_LC = Dropout(0.2)(concat_LC)

    TimeD = TimeDistributed(Dense(targetvocabsize + 1))(concat_LC)

    model = Activation('softmax')(TimeD)

    # crf = CRF(targetvocabsize+1, sparse_target=False)
    # model = crf(TimeD)

    Models = Model([word_input, posl_input, entl_input], model)

    Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    # Models.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])

    return Models


def creat_Model_BiLSTM_CRF(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim,
                              sourcecharsize, character_W, input_word_length, char_emd_dim,
                              sourcepossize, pos_W,pos_emd_dim,
                              loss='categorical_crossentropy', optimizer='rmsprop'):

    word_input = Input(shape=(input_seq_lenth,), dtype='int32')

    char_input = Input(shape=(input_seq_lenth, input_word_length,), dtype='int32')

    char_embedding = Embedding(input_dim= sourcecharsize,
                               output_dim=char_emd_dim,
                               batch_input_shape=(batch_size, input_seq_lenth, input_word_length),
                               mask_zero=True,
                               trainable=True,
                               weights=[character_W])

    char_embedding2 = TimeDistributed(char_embedding)(char_input)

    char_cnn = TimeDistributed(Conv1D(50, 3, activation='relu', padding='same'))(char_embedding2)

    char_macpool = TimeDistributed(GlobalMaxPooling1D())(char_cnn)
    char_macpool = Dropout(0.5)(char_macpool)

    word_embedding = Embedding(input_dim=sourcevocabsize + 1,
                              output_dim=emd_dim,
                              input_length=input_seq_lenth,
                              mask_zero=True,
                              trainable=True,
                              weights=[source_W])(word_input)


    word_embedding_dropout = Dropout(0.5)(word_embedding)

    pos_input = Input(shape=(input_seq_lenth,), dtype='int32')
    pos_embeding = Embedding(input_dim=sourcepossize + 1,
                                  output_dim=pos_emd_dim,
                                  input_length=input_seq_lenth,
                                  mask_zero=False,
                                  trainable=True,
                                  weights=[pos_W])(pos_input)

    embedding = concatenate([word_embedding_dropout, char_macpool], axis=-1)

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode = 'concat')(embedding)
    # BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True))(word_embedding_dropout)
    BiLSTM_dropout = Dropout(0.5)(BiLSTM)

    BiLSTM_dropout = BatchNormalization(axis=1)(BiLSTM_dropout)

    TimeD = TimeDistributed(Dense(targetvocabsize+1))(BiLSTM_dropout)
    # TimeD = TimeDistributed(Dense(int(hidden_dim / 2)))(BiLSTM_dropout)
    TimeD = Dropout(0.5)(TimeD)

    # model = Activation('softmax')(TimeD)

    crflayer = CRF(targetvocabsize+1, sparse_target=False)
    model = crflayer(TimeD)#0.8746633147782367
    # # model = crf(BiLSTM_dropout)#0.870420501714492

    Models = Model([word_input, char_input], model)

    # Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    Models.compile(loss=crflayer.loss_function, optimizer='adam', metrics=[crflayer.accuracy])
    # Models.compile(loss=crf.loss_function, optimizer=optimizers.RMSprop(lr=0.01), metrics=[crf.accuracy])

    return Models


def Model_BiLSTM_CRF_multi2(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim,
                              sourcecharsize, character_W, input_word_length, char_emd_dim,
                              sourcepossize, pos_W,pos_emd_dim,
                              loss='categorical_crossentropy', optimizer='rmsprop'):

    word_input = Input(shape=(input_seq_lenth,), dtype='int32')
    char_input = Input(shape=(input_seq_lenth, input_word_length,), dtype='int32')

    char_embedding = Embedding(input_dim= sourcecharsize,
                               output_dim=char_emd_dim,
                               batch_input_shape=(batch_size, input_seq_lenth, input_word_length),
                               mask_zero=True,
                               trainable=True,
                               weights=[character_W])
    char_embedding2 = TimeDistributed(char_embedding)(char_input)
    char_cnn = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))(char_embedding2)
    char_macpool = TimeDistributed(GlobalMaxPooling1D())(char_cnn)
    char_macpool = Dropout(0.5)(char_macpool)

    word_embedding = Embedding(input_dim=sourcevocabsize + 1,
                              output_dim=emd_dim,
                              input_length=input_seq_lenth,
                              mask_zero=True,
                              trainable=True,
                              weights=[source_W])(word_input)
    word_embedding_dropout = Dropout(0.5)(word_embedding)

    pos_input = Input(shape=(input_seq_lenth,), dtype='int32')
    pos_embeding = Embedding(input_dim=sourcepossize + 1,
                                  output_dim=pos_emd_dim,
                                  input_length=input_seq_lenth,
                                  mask_zero=False,
                                  trainable=True,
                                  weights=[pos_W])(pos_input)

    embedding = concatenate([word_embedding_dropout, char_macpool], axis=-1)

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode = 'concat')(embedding)
    # BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True))(word_embedding_dropout)
    BiLSTM_dropout = Dropout(0.5)(BiLSTM)

    BiLSTM_dropout = BatchNormalization(axis=1)(BiLSTM_dropout)


    mlp2_hidden1 = TimeDistributed(Dense(50, activation='relu'))(BiLSTM_dropout)
    output1 = TimeDistributed(Dense(2+1, activation='sigmoid', name='isOther'))(mlp2_hidden1)

    mlp1_hidden1 =TimeDistributed(Dense(100, activation='relu'))(BiLSTM_dropout)
    mlp1_concat = concatenate([mlp2_hidden1, mlp1_hidden1], axis=-1)
    mlp1_hidden2 = TimeDistributed(Dense(targetvocabsize+1, activation='relu'))(mlp1_concat)
    crflayer = CRF(targetvocabsize+1, sparse_target=False, name='finall')
    output12 = crflayer(mlp1_hidden2)

    Models = Model([word_input, char_input], [output12, output1])
    Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
                   loss={'finall': crflayer.loss_function, 'isOther': 'binary_crossentropy'},
                   loss_weights={'finall': 1., 'isOther': 1.},
                   metrics={'finall': [crflayer.accuracy], 'isOther': ['acc']})

    return Models


def test_model(nn_model, testdata, chardata, pos_data, index2word, resultfile='', batch_size=50):
    index2word[0] = ''
    testx = np.asarray(testdata[0], dtype="int32")
    testy = np.asarray(testdata[1], dtype="int32")
    poslabel_test = np.asarray(pos_data, dtype="int32")
    testchar = np.asarray(chardata, dtype="int32")

    testresult = []
    testresult2 = []
    predictions = nn_model.predict([testx, testchar])


    if len(predictions) >= 2:

        for si in range(0, len(predictions[0])):

            sent = predictions[0][si]
            ptag = []
            for word in sent:
                next_index = np.argmax(word)
                next_token = index2word[next_index]
                ptag.append(next_token)
            # print('next_token--ptag--',str(ptag))

            sent = predictions[1][si]
            ptag2 = []
            for word in sent:
                next_index = np.argmax(word)
                next_token = index2word[next_index]
                ptag2.append(next_token)
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

            result2 = []
            result2.append(ptag2)
            result2.append(ttag)
            testresult2.append(result2)

        P, R, F, PR_count, P_count, TR_count = evaluation_NER(testresult2)
        print('OP2>>>>>>>>>>')
        print(P, R, F)

    else:
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

    pickle.dump(testresult, open(resultfile, 'wb'))

    P, R, F, PR_count, P_count, TR_count = evaluation_NER(testresult)

    return P, R, F, PR_count, P_count, TR_count



def SelectModel(modelname, sourcevocabsize, targetvocabsize, source_W,
                             input_seq_lenth,
                             output_seq_lenth,
                             hidden_dim, emd_dim,
                     sourcecharsize,character_W,input_word_length,char_emd_dim,
                        sourcepossize, pos_W,pos_emd_dim,
                     loss='categorical_crossentropy', optimizer='rmsprop'):
    nn_model = None
    if modelname is 'creat_Model_BiLSTM_CnnDecoder':
        nn_model = creat_Model_BiLSTM_CnnDecoder(sourcevocabsize=sourcevocabsize, targetvocabsize=targetvocabsize,
                                                       source_W=source_W,
                                                       input_seq_lenth=input_seq_lenth,
                                                       output_seq_lenth=output_seq_lenth,
                                                       hidden_dim=hidden_dim, emd_dim=emd_dim,
                                                        sourcecharsize=sourcecharsize,
                                                        character_W=character_W,
                                                        input_word_length=input_word_length,
                                                        char_emd_dim=char_emd_dim,
                                                 sourcepossize=sourcepossize, pos_W=pos_W, pos_emd_dim=pos_emd_dim)

    elif modelname is 'creat_Model_BiLSTM_CRF':
        nn_model = creat_Model_BiLSTM_CRF(sourcevocabsize=sourcevocabsize, targetvocabsize=targetvocabsize,
                                                       source_W=source_W,
                                                       input_seq_lenth=input_seq_lenth,
                                                       output_seq_lenth=output_seq_lenth,
                                                       hidden_dim=hidden_dim, emd_dim=emd_dim,
                                                        sourcecharsize=sourcecharsize,
                                                        character_W=character_W,
                                                        input_word_length=input_word_length,
                                                        char_emd_dim=char_emd_dim,
                                          sourcepossize=sourcepossize, pos_W=pos_W, pos_emd_dim=pos_emd_dim)
    elif modelname is 'Model_BiLSTM_CRF_multi2':
        nn_model = Model_BiLSTM_CRF_multi2(sourcevocabsize=sourcevocabsize, targetvocabsize=targetvocabsize,
                                                       source_W=source_W,
                                                       input_seq_lenth=input_seq_lenth,
                                                       output_seq_lenth=output_seq_lenth,
                                                       hidden_dim=hidden_dim, emd_dim=emd_dim,
                                                        sourcecharsize=sourcecharsize,
                                                        character_W=character_W,
                                                        input_word_length=input_word_length,
                                                        char_emd_dim=char_emd_dim,
                                          sourcepossize=sourcepossize, pos_W=pos_W, pos_emd_dim=pos_emd_dim)

    elif modelname is 'Model_BiLSTM_CnnDecoder_multi2':
        nn_model = Model_BiLSTM_CnnDecoder_multi2(sourcevocabsize=sourcevocabsize, targetvocabsize=targetvocabsize,
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
    y_O = np.asarray(traindata[2], dtype="int32")
    y_BIOES = np.asarray(traindata[3], dtype="int32")
    # entlabel_train = np.asarray(entlabel_traindata, dtype="int32")
    # poslabel_train = np.asarray(poslabel_traindata, dtype="int32")
    input_char = np.asarray(chartrain, dtype="int32")
    x_word_val = np.asarray(devdata[0], dtype="int32")
    y_val = np.asarray(devdata[1], dtype="int32")
    y_O_val = np.asarray(devdata[2], dtype="int32")
    y_BIOES_val = np.asarray(devdata[3], dtype="int32")
    input_char_val = np.asarray(chardev, dtype="int32")

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
        history = nn_model.fit([x_word, input_char], [y, y_O, y_BIOES],
                               batch_size=batch_size,
                               epochs=1,
                               validation_data=([x_word_val, input_char_val], [y_val, y_O_val, y_BIOES_val]),
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
            resultfile = resultdir+"result-"+str(saveepoch)
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

            print(epoch, P, R, F, '  maxF=',maxF)

        if earlystopping >= 10:
            break

    return nn_model


def getClass_weight(x=10):
    cw = {0: 1, 1: 1}
    for i in range(2, x+1):
        cw[i] = 10
    return cw

def infer_e2e_model(modelname, datafile, lstm_modelfile, resultdir, hidden_dim=200, batch_size=50):
    # traindata, testdata, source_W, source_vob, sourc_idex_word, target_vob, \
    # target_idex_word, max_s, k \
    #     = pickle.load(open(eelstmfile, 'rb'))

    # load training data and test data

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
    resultfile = resultdir + "result-" + 'infer_test'

    P, R, F, PR_count, P_count, TR_count = test_model(nnmodel, testdata, chartest,pos_test, target_idex_word, resultfile,
                                                      batch_size)
    print('P= ', P, '  R= ', R, '  F= ', F)


if __name__ == "__main__":

    maxlen = 50


    modelname = 'creat_Model_BiLSTM_CRF'
    modelname = 'Model_BiLSTM_CRF_multi2'
    modelname = 'Model_BiLSTM_CnnDecoder_multi2'
    # modelname = 'creat_Model_BiLSTM_CnnDecoder'

    print(modelname)

    w2v_file = "./data/w2v/glove.6B.100d.txt"
    datafile = "./data/model/data_fix_multi3.pkl"
    # modelfile = "./data/model/BiLSTM_CnnDecoder_wordFixCharembed_model3.h5"
    modelfile = "./data/model/" + modelname + "_41.h5"

    resultdir = "./data/result/"

    trainfile = "./data/CoNLL2003_NER/eng.train.BIOES.txt"
    devfile = "./data/CoNLL2003_NER/eng.testa.BIOES.txt"
    testfile = "./data/CoNLL2003_NER/eng.testb.BIOES.txt"

    batch_size = 100
    retrain = False
    Test = True
    valid = False
    Label = False
    if not os.path.exists(datafile):
        print("Precess data....")
        get_data(trainfile,devfile, testfile, w2v_file, datafile, w2v_k=100, char_emd_dim=25, maxlen=maxlen)
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
        print(modelfile)
        infer_e2e_model(modelname, datafile, modelfile, resultdir, hidden_dim=200, batch_size=batch_size)





    '''
    lstm hidenlayer,
    bash size,
    epoach
    '''
# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
#
# KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))

# CUDA_VISIBLE_DEVICES="" python Model.py
