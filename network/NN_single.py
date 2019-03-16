# -*- encoding:utf-8 -*-


import pickle, datetime, codecs
import os.path
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from PrecessData import get_data, make_idx_data_index
from Evaluate import evaluation_NER, evaluation_NER2, evaluation_NER_BIOES,evaluation_NER_Type
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


def Model_BiLSTM_CRF(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim,
                              sourcecharsize, character_W, input_word_length, char_emd_dim,
                              batch_size=32, loss='categorical_crossentropy', optimizer='rmsprop'):

    word_input = Input(shape=(input_seq_lenth,), dtype='int32')

    char_input = Input(shape=(input_seq_lenth, input_word_length,), dtype='int32')

    char_embedding = Embedding(input_dim= sourcecharsize,
                               output_dim=char_emd_dim,
                               batch_input_shape=(batch_size, input_seq_lenth, input_word_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[character_W])

    char_embedding2 = TimeDistributed(char_embedding)(char_input)

    char_cnn = TimeDistributed(Conv1D(50, 3, activation='relu', padding='same'))(char_embedding2)

    char_macpool = TimeDistributed(GlobalMaxPooling1D())(char_cnn)
    # char_macpool = Dropout(0.5)(char_macpool)
    # !!!!!!!!!!!!!!
    char_macpool = Dropout(0.25)(char_macpool)

    word_embedding = Embedding(input_dim=sourcevocabsize + 1,
                              output_dim=emd_dim,
                              input_length=input_seq_lenth,
                              mask_zero=True,
                              trainable=True,
                              weights=[source_W])(word_input)


    word_embedding_dropout = Dropout(0.5)(word_embedding)

    embedding = concatenate([word_embedding_dropout, char_macpool], axis=-1)

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(embedding)
    BiLSTM = BatchNormalization(axis=1)(BiLSTM)
    BiLSTM_dropout = Dropout(0.5)(BiLSTM)

    TimeD = TimeDistributed(Dense(targetvocabsize+1))(BiLSTM_dropout)
    # TimeD = TimeDistributed(Dense(int(hidden_dim / 2)))(BiLSTM_dropout)
    # TimeD = Dropout(0.5)(TimeD)
    # !!!!!!!!!!!!!!!delete dropout

    # model = Activation('softmax')(TimeD)

    crflayer = CRF(targetvocabsize+1, sparse_target=False)
    model = crflayer(TimeD)#0.8746633147782367
    # # model = crf(BiLSTM_dropout)#0.870420501714492

    Models = Model([word_input, char_input], [model])

    # Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    # Models.compile(loss=crflayer.loss_function, optimizer='adam', metrics=[crflayer.accuracy])
    Models.compile(loss=crflayer.loss_function, optimizer=optimizers.RMSprop(lr=0.001), metrics=[crflayer.accuracy])

    return Models


def Model_BiLSTM_X2_CRF(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim,
                              sourcecharsize, character_W, input_word_length, char_emd_dim,
                              batch_size=32, loss='categorical_crossentropy', optimizer='rmsprop'):

    word_input = Input(shape=(input_seq_lenth,), dtype='int32')

    char_input = Input(shape=(input_seq_lenth, input_word_length,), dtype='int32')

    char_embedding = Embedding(input_dim= sourcecharsize,
                               output_dim=char_emd_dim,
                               batch_input_shape=(batch_size, input_seq_lenth, input_word_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[character_W])

    char_embedding2 = TimeDistributed(char_embedding)(char_input)

    char_cnn = TimeDistributed(Conv1D(50, 3, activation='relu', padding='same'))(char_embedding2)

    char_macpool = TimeDistributed(GlobalMaxPooling1D())(char_cnn)
    # char_macpool = Dropout(0.5)(char_macpool)
    # !!!!!!!!!!!!!!
    char_macpool = Dropout(0.25)(char_macpool)

    word_embedding = Embedding(input_dim=sourcevocabsize + 1,
                              output_dim=emd_dim,
                              input_length=input_seq_lenth,
                              mask_zero=True,
                              trainable=True,
                              weights=[source_W])(word_input)


    word_embedding_dropout = Dropout(0.5)(word_embedding)

    embedding = concatenate([word_embedding_dropout, char_macpool], axis=-1)

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(embedding)
    BiLSTM = BatchNormalization(axis=1)(BiLSTM)
    BiLSTM_dropout = Dropout(0.5)(BiLSTM)

    BiLSTM2 = Bidirectional(LSTM(hidden_dim // 2, return_sequences=True), merge_mode='concat')(BiLSTM_dropout)
    BiLSTM_dropout2 = Dropout(0.5)(BiLSTM2)

    TimeD = TimeDistributed(Dense(targetvocabsize+1))(BiLSTM_dropout2)
    # TimeD = TimeDistributed(Dense(int(hidden_dim / 2)))(BiLSTM_dropout)
    # TimeD = Dropout(0.5)(TimeD)
    # !!!!!!!!!!!!!!!delete dropout

    # model = Activation('softmax')(TimeD)

    crflayer = CRF(targetvocabsize+1, sparse_target=False)
    model = crflayer(TimeD)#0.8746633147782367
    # # model = crf(BiLSTM_dropout)#0.870420501714492

    Models = Model([word_input, char_input], [model])

    # Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    # Models.compile(loss=crflayer.loss_function, optimizer='adam', metrics=[crflayer.accuracy])
    Models.compile(loss=crflayer.loss_function, optimizer=optimizers.RMSprop(lr=0.001), metrics=[crflayer.accuracy])

    return Models


def Model_BiLSTM_Softmax(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim,
                              sourcecharsize, character_W, input_word_length, char_emd_dim,
                              sourcepossize, pos_W,pos_emd_dim, batch_size=32,
                              loss='categorical_crossentropy', optimizer='rmsprop'):

    word_input = Input(shape=(input_seq_lenth,), dtype='int32')

    char_input = Input(shape=(input_seq_lenth, input_word_length,), dtype='int32')

    char_embedding = Embedding(input_dim= sourcecharsize,
                               output_dim=char_emd_dim,
                               batch_input_shape=(batch_size, input_seq_lenth, input_word_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[character_W])

    char_embedding2 = TimeDistributed(char_embedding)(char_input)

    char_cnn = TimeDistributed(Conv1D(50, 3, activation='relu', padding='same'))(char_embedding2)

    char_macpool = TimeDistributed(GlobalMaxPooling1D())(char_cnn)
    # char_macpool = Dropout(0.5)(char_macpool)
    # !!!!!!!!!!!!!!
    char_macpool = Dropout(0.25)(char_macpool)

    word_embedding = Embedding(input_dim=sourcevocabsize + 1,
                              output_dim=emd_dim,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=True,
                              weights=[source_W])(word_input)


    word_embedding_dropout = Dropout(0.5)(word_embedding)

    embedding = concatenate([word_embedding_dropout, char_macpool], axis=-1)

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode = 'concat')(embedding)
    # BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True))(word_embedding_dropout)
    BiLSTM = BatchNormalization(axis=1)(BiLSTM)
    BiLSTM_dropout = Dropout(0.5)(BiLSTM)

    TimeD = TimeDistributed(Dense(targetvocabsize+1))(BiLSTM_dropout)
    # TimeD = TimeDistributed(Dense(int(hidden_dim / 2)))(BiLSTM_dropout)
    # TimeD = Dropout(0.5)(TimeD)
    # !!!!!!!!!!!!!!!delete dropout

    model = Activation('softmax')(TimeD)

    # crflayer = CRF(targetvocabsize+1, sparse_target=False)
    # model = crflayer(TimeD)


    Models = Model([word_input, char_input], [model])

    Models.compile(loss=loss, optimizer=optimizers.RMSprop(lr=0.001), metrics=['acc'])

    # Models.compile(loss=crflayer.loss_function, optimizer=optimizers.RMSprop(lr=0.001), metrics=[crflayer.accuracy])

    return Models


def Model_BiLSTM_parallel_8_64_CRF(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim,
                              sourcecharsize, character_W, input_word_length, char_emd_dim,
                              sourcepossize, pos_W,pos_emd_dim, batch_size=32,
                              loss='categorical_crossentropy', optimizer='rmsprop'):

    word_input = Input(shape=(input_seq_lenth,), dtype='int32')

    char_input = Input(shape=(input_seq_lenth, input_word_length,), dtype='int32')

    char_embedding = Embedding(input_dim= sourcecharsize,
                               output_dim=char_emd_dim,
                               batch_input_shape=(batch_size, input_seq_lenth, input_word_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[character_W])

    char_embedding2 = TimeDistributed(char_embedding)(char_input)

    char_cnn = TimeDistributed(Conv1D(50, 3, activation='relu', padding='same'))(char_embedding2)

    char_macpool = TimeDistributed(GlobalMaxPooling1D())(char_cnn)
    char_macpool = Dropout(0.5)(char_macpool)

    word_embedding = Embedding(input_dim=sourcevocabsize + 1,
                              output_dim=emd_dim,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=True,
                              weights=[source_W])(word_input)


    word_embedding_dropout = Dropout(0.5)(word_embedding)

    embedding = concatenate([word_embedding_dropout, char_macpool], axis=-1)

    BiLSTMlist = []

    for i in range(4):
        BiLSTM_i = Bidirectional(LSTM(50, return_sequences=True), merge_mode='concat')(embedding)
        BiLSTM_i = Dropout(0.5)(BiLSTM_i)
        BiLSTMlist.append(BiLSTM_i)

    decoder_input = concatenate(BiLSTMlist, axis=-1)

    TimeD = TimeDistributed(Dense(targetvocabsize+1))(decoder_input)
    # TimeD = TimeDistributed(Dense(int(hidden_dim / 2)))(BiLSTM_dropout)
    TimeD = Dropout(0.5)(TimeD)

    # model = Activation('softmax')(TimeD)

    crflayer = CRF(targetvocabsize+1, sparse_target=False)
    model = crflayer(TimeD)#0.8746633147782367
    # # model = crf(BiLSTM_dropout)#0.870420501714492

    Models = Model([word_input, char_input], [model])

    # Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    # Models.compile(loss=crflayer.loss_function, optimizer='adam', metrics=[crflayer.accuracy])
    Models.compile(loss=crflayer.loss_function, optimizer=optimizers.RMSprop(lr=0.001), metrics=[crflayer.accuracy])

    return Models


def Model_BiLSTM_CRF_withPOS(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim,
                              sourcecharsize, character_W, input_word_length, char_emd_dim,
                              sourcepossize, pos_W, pos_emd_dim, batch_size=32, pos_k=3,
                              loss='categorical_crossentropy', optimizer='rmsprop'):

    word_input = Input(shape=(input_seq_lenth,), dtype='int32')

    char_input = Input(shape=(input_seq_lenth, input_word_length,), dtype='int32')

    char_embedding = Embedding(input_dim= sourcecharsize,
                               output_dim=char_emd_dim,
                               batch_input_shape=(batch_size, input_seq_lenth, input_word_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[character_W])

    char_embedding2 = TimeDistributed(char_embedding)(char_input)

    char_cnn = TimeDistributed(Conv1D(50, 3, activation='relu', padding='same'))(char_embedding2)

    char_macpool = TimeDistributed(GlobalMaxPooling1D())(char_cnn)
    char_macpool = Dropout(0.5)(char_macpool)

    word_embedding = Embedding(input_dim=sourcevocabsize + 1,
                              output_dim=emd_dim,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=True,
                              weights=[source_W])(word_input)


    word_embedding_dropout = Dropout(0.5)(word_embedding)

    pos_input = Input(shape=(input_seq_lenth, pos_k,), dtype='int32')
    pos_embedding = Embedding(input_dim=sourcepossize + 1,
                              output_dim=pos_emd_dim,
                              batch_input_shape=(batch_size, input_seq_lenth, pos_k),
                              mask_zero=False,
                              trainable=True,
                              weights=[pos_W])
    pos_embedding2 = TimeDistributed(pos_embedding)(pos_input)
    pos_cnn = TimeDistributed(Conv1D(50, pos_k, activation='relu', border_mode='valid'))(pos_embedding2)
    pos_macpool = TimeDistributed(GlobalMaxPooling1D())(pos_cnn)

    pos_rnn = TimeDistributed(Bidirectional(LSTM(25, return_sequences=False), merge_mode='concat'))(pos_embedding2)


    embedding = concatenate([word_embedding_dropout, char_macpool, pos_rnn], axis=-1)

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode = 'concat')(embedding)
    # BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True))(word_embedding_dropout)
    BiLSTM = BatchNormalization(axis=1)(BiLSTM)
    BiLSTM_dropout = Dropout(0.5)(BiLSTM)

    TimeD = TimeDistributed(Dense(targetvocabsize+1))(BiLSTM_dropout)
    # TimeD = TimeDistributed(Dense(int(hidden_dim / 2)))(BiLSTM_dropout)
    TimeD = Dropout(0.5)(TimeD)

    # model = Activation('softmax')(TimeD)

    crflayer = CRF(targetvocabsize+1, sparse_target=False)
    model = crflayer(TimeD)#0.8746633147782367
    # # model = crf(BiLSTM_dropout)#0.870420501714492

    Models = Model([word_input, char_input, pos_input], [model])

    # Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    # Models.compile(loss=crflayer.loss_function, optimizer='adam', metrics=[crflayer.accuracy])
    Models.compile(loss=crflayer.loss_function, optimizer=optimizers.RMSprop(lr=0.001), metrics=[crflayer.accuracy])

    return Models


def Model_BiLSTM_CnnDecoder(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim,
                              sourcecharsize, character_W, input_word_length, char_emd_dim,
                              sourcepossize, pos_W, pos_emd_dim,batch_size=32,
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


def Model_Dense_Softmax(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim,
                              sourcecharsize, character_W, input_word_length, char_emd_dim,
                              sourcepossize, pos_W,pos_emd_dim, batch_size=32,
                              loss='categorical_crossentropy', optimizer='rmsprop'):

    word_input = Input(shape=(input_seq_lenth,), dtype='int32')

    char_input = Input(shape=(input_seq_lenth, input_word_length,), dtype='int32')

    char_embedding = Embedding(input_dim= sourcecharsize,
                               output_dim=char_emd_dim,
                               batch_input_shape=(batch_size, input_seq_lenth, input_word_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[character_W])

    char_embedding2 = TimeDistributed(char_embedding)(char_input)

    char_cnn = TimeDistributed(Conv1D(50, 3, activation='relu', padding='same'))(char_embedding2)

    char_macpool = TimeDistributed(GlobalMaxPooling1D())(char_cnn)
    # char_macpool = Dropout(0.5)(char_macpool)
    # !!!!!!!!!!!!!!
    char_macpool = Dropout(0.25)(char_macpool)

    word_embedding = Embedding(input_dim=sourcevocabsize + 1,
                              output_dim=emd_dim,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=True,
                              weights=[source_W])(word_input)


    word_embedding_dropout = Dropout(0.5)(word_embedding)

    embedding = concatenate([word_embedding_dropout, char_macpool], axis=-1)

    Dense1 = TimeDistributed(Dense(400, activation='tanh'))(embedding)
    Dense1 = Dropout(0.5)(Dense1)
    Dense2 = TimeDistributed(Dense(200, activation='tanh'))(Dense1)
    Dense2 = Dropout(0.3)(Dense2)
    Dense3 = TimeDistributed(Dense(100, activation='tanh'))(Dense2)
    Dense3 = Dropout(0.2)(Dense3)

    TimeD = TimeDistributed(Dense(targetvocabsize+1))(Dense3)

    # TimeD = Dropout(0.5)(TimeD)
    # !!!!!!!!!!!!!!!delete dropout

    model = Activation('softmax')(TimeD)

    # crflayer = CRF(targetvocabsize+1, sparse_target=False)
    # model = crflayer(TimeD)


    Models = Model([word_input, char_input], [model])

    Models.compile(loss=loss, optimizer=optimizers.RMSprop(lr=0.001), metrics=['acc'])

    # Models.compile(loss=crflayer.loss_function, optimizer=optimizers.RMSprop(lr=0.001), metrics=[crflayer.accuracy])

    return Models


if __name__ == '__main__':
    batch_size = 32