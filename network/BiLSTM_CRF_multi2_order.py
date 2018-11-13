# -*- encoding:utf-8 -*-


import pickle, datetime, codecs
import os.path
import numpy as np
from keras.layers import Flatten,Lambda,Conv2D
from keras.layers.core import Dropout, Activation, Permute, RepeatVector
from keras.layers.merge import concatenate, Concatenate, multiply, Dot, average
from keras.layers import TimeDistributed, Input, Bidirectional, Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D, RepeatVector, AveragePooling1D
from keras.models import Model
from keras_contrib.layers import CRF
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback


def BiLSTM_CRF_multi2_order(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim,
                              sourcecharsize, character_W, input_word_length, char_emd_dim,
                              sourcepossize, pos_W,pos_emd_dim,
                            batch_size=32,
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
    char_cnn = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))(char_embedding2)
    char_macpool = TimeDistributed(GlobalMaxPooling1D())(char_cnn)
    char_macpool = Dropout(0.5)(char_macpool)

    word_embedding = Embedding(input_dim=sourcevocabsize + 1,
                              output_dim=emd_dim,
                              input_length=input_seq_lenth,
                              mask_zero=False,
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

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True,), merge_mode = 'concat')(embedding)
    # BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True))(word_embedding_dropout)
    BiLSTM = BatchNormalization(axis=1)(BiLSTM)
    BiLSTM_dropout = Dropout(0.5)(BiLSTM)

    mlp1_hidden1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(BiLSTM_dropout)
    mlp1_hidden1 = Dropout(0.5)(mlp1_hidden1)
    mlp1_hidden2 = TimeDistributed(Dense(100, activation='relu'))(mlp1_hidden1)
    # output1 = TimeDistributed(Dense(5+1, activation='softmax'), name='BIOES')(decodelayer1)
    mlp1_hidden3 = TimeDistributed(Dense(5 + 1, activation=None))(mlp1_hidden2)
    crflayer1 = CRF(5 + 1, sparse_target=False, learn_mode='marginal', name='BIOES')
    output1 = crflayer1(mlp1_hidden3)

    mlp2_hidden1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(BiLSTM_dropout)
    mlp2_hidden1 = Dropout(0.5)(mlp2_hidden1)
    mlp2_hidden2 = TimeDistributed(Dense(100, activation='relu'))(mlp2_hidden1)
    mlp2_hidden3 = concatenate([mlp1_hidden2, mlp2_hidden2], axis=-1)
    mlp2_hidden3 = TimeDistributed(Dense(100, activation='tanh'))(mlp2_hidden3)
    # output2 = TimeDistributed(Dense(5+1, activation='softmax'), name='Type')(decodelayer2)
    mlp2_hidden3 = TimeDistributed(Dense(5 + 1, activation=None))(mlp2_hidden3)
    crflayer2 = CRF(5 + 1, sparse_target=False, name='Type', learn_mode='marginal')
    output2 = crflayer2(mlp2_hidden3)


    Models = Model([word_input, char_input], [output1, output2])
    # Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
    #                loss={'finall': crflayer.loss_function, 'BIOES': 'categorical_crossentropy', 'Type': 'categorical_crossentropy'},
    #                loss_weights={'finall': 1., 'BIOES': 1., 'Type': 1.},
    #                metrics={'finall': [crflayer.accuracy], 'BIOES': ['acc'], 'Type': ['acc']})

    Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
                   loss={'BIOES': crflayer1.loss_function, 'Type': crflayer2.loss_function},
                   loss_weights={'BIOES': 1., 'Type': 1.},
                   metrics={'BIOES': [crflayer2.accuracy], 'Type': [crflayer2.accuracy]})

    return Models

def BiLSTM_CRF_multi2_order2(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim,
                              sourcecharsize, character_W, input_word_length, char_emd_dim,
                              sourcepossize, pos_W,pos_emd_dim,
                            batch_size=32,
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
    char_cnn = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))(char_embedding2)
    char_macpool = TimeDistributed(GlobalMaxPooling1D())(char_cnn)
    char_macpool = Dropout(0.5)(char_macpool)

    word_embedding = Embedding(input_dim=sourcevocabsize + 1,
                              output_dim=emd_dim,
                              input_length=input_seq_lenth,
                              mask_zero=False,
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

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True,), merge_mode = 'concat')(embedding)
    # BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True))(word_embedding_dropout)
    BiLSTM = BatchNormalization(axis=1)(BiLSTM)
    BiLSTM_dropout = Dropout(0.5)(BiLSTM)

    mlp1_hidden1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(BiLSTM_dropout)
    mlp1_hidden1 = Dropout(0.5)(mlp1_hidden1)
    mlp1_hidden2 = TimeDistributed(Dense(100, activation='relu'))(mlp1_hidden1)
    mlp1_hidden2 = TimeDistributed(Dense(100, activation='tanh'))(mlp1_hidden1)
    # output1 = TimeDistributed(Dense(5+1, activation='softmax'), name='BIOES')(decodelayer1)
    mlp1_hidden3 = TimeDistributed(Dense(5 + 1, activation=None))(mlp1_hidden2)
    crflayer1 = CRF(5 + 1, sparse_target=False, learn_mode='marginal', name='BIOES')
    output1 = crflayer1(mlp1_hidden3)

    mlp2_hidden1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(BiLSTM_dropout)
    mlp2_hidden1 = Dropout(0.5)(mlp2_hidden1)
    mlp2_hidden2 = TimeDistributed(Dense(100, activation='relu'))(mlp2_hidden1)
    mlp2_hidden2 = TimeDistributed(Dense(100, activation='tanh'))(mlp2_hidden1)
    mlp2_hidden3 = concatenate([mlp1_hidden2, mlp2_hidden2], axis=-1)
    mlp2_hidden3 = TimeDistributed(Dense(100, activation='tanh'))(mlp2_hidden3)
    # output2 = TimeDistributed(Dense(5+1, activation='softmax'), name='Type')(decodelayer2)
    mlp2_hidden3 = TimeDistributed(Dense(targetvocabsize + 1, activation=None))(mlp2_hidden3)
    crflayer2 = CRF(targetvocabsize + 1, sparse_target=False, name='Type', learn_mode='marginal')
    output2 = crflayer2(mlp2_hidden3)


    Models = Model([word_input, char_input], [output1, output2])
    # Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
    #                loss={'finall': crflayer.loss_function, 'BIOES': 'categorical_crossentropy', 'Type': 'categorical_crossentropy'},
    #                loss_weights={'finall': 1., 'BIOES': 1., 'Type': 1.},
    #                metrics={'finall': [crflayer.accuracy], 'BIOES': ['acc'], 'Type': ['acc']})

    Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
                   loss={'BIOES': crflayer1.loss_function, 'Type': crflayer2.loss_function},
                   loss_weights={'BIOES': 1., 'Type': 1.},
                   metrics={'BIOES': [crflayer2.accuracy], 'Type': [crflayer2.accuracy]})

    return Models


def BiLSTM_CRF_multi2_order3(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim,
                              sourcecharsize, character_W, input_word_length, char_emd_dim,
                              sourcepossize, pos_W,pos_emd_dim,
                            batch_size=32,
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
    char_cnn = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))(char_embedding2)
    char_macpool = TimeDistributed(GlobalMaxPooling1D())(char_cnn)
    char_macpool = Dropout(0.5)(char_macpool)

    word_embedding = Embedding(input_dim=sourcevocabsize + 1,
                              output_dim=emd_dim,
                              input_length=input_seq_lenth,
                              mask_zero=False,
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

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True,), merge_mode = 'concat')(embedding)
    # BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True))(word_embedding_dropout)
    BiLSTM = BatchNormalization(axis=1)(BiLSTM)
    BiLSTM_dropout = Dropout(0.5)(BiLSTM)

    mlp1_hidden1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(BiLSTM_dropout)
    mlp1_hidden1 = Dropout(0.5)(mlp1_hidden1)
    mlp1_hidden2 = TimeDistributed(Dense(100, activation='tanh'))(mlp1_hidden1)
    # output1 = TimeDistributed(Dense(5+1, activation='softmax'), name='BIOES')(decodelayer1)
    mlp1_hidden3 = TimeDistributed(Dense(5 + 1, activation=None))(mlp1_hidden2)
    crflayer1 = CRF(5 + 1, sparse_target=False, learn_mode='marginal', name='BIOES')
    output1 = crflayer1(mlp1_hidden3)

    mlp2_hidden1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(BiLSTM_dropout)
    mlp2_hidden1 = Dropout(0.5)(mlp2_hidden1)
    mlp2_hidden2 = TimeDistributed(Dense(100, activation='tanh'))(mlp2_hidden1)
    mlp2_hidden3 = TimeDistributed(Dense(100, activation='tanh'))(mlp2_hidden2)
    # output2 = TimeDistributed(Dense(5+1, activation='softmax'), name='Type')(decodelayer2)
    mlp2_hidden3 = TimeDistributed(Dense(5 + 1, activation=None))(mlp2_hidden3)
    mlp2_hidden4 = average([mlp1_hidden3, mlp2_hidden3])
    crflayer2 = CRF(5 + 1, sparse_target=False, name='Type', learn_mode='marginal')
    output2 = crflayer2(mlp2_hidden4)


    Models = Model([word_input, char_input], [output1, output2])
    # Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
    #                loss={'finall': crflayer.loss_function, 'BIOES': 'categorical_crossentropy', 'Type': 'categorical_crossentropy'},
    #                loss_weights={'finall': 1., 'BIOES': 1., 'Type': 1.},
    #                metrics={'finall': [crflayer.accuracy], 'BIOES': ['acc'], 'Type': ['acc']})

    Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
                   loss={'BIOES': crflayer1.loss_function, 'Type': crflayer2.loss_function},
                   loss_weights={'BIOES': 1., 'Type': 1.},
                   metrics={'BIOES': [crflayer2.accuracy], 'Type': [crflayer2.accuracy]})

    return Models


def BiLSTM_CRF_multi2_order4(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim,
                              sourcecharsize, character_W, input_word_length, char_emd_dim,
                              sourcepossize, pos_W,pos_emd_dim,
                            batch_size=32,
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
    char_cnn = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))(char_embedding2)
    char_macpool = TimeDistributed(GlobalMaxPooling1D())(char_cnn)
    char_macpool = Dropout(0.5)(char_macpool)

    word_embedding = Embedding(input_dim=sourcevocabsize + 1,
                              output_dim=emd_dim,
                              input_length=input_seq_lenth,
                              mask_zero=False,
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

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True,), merge_mode = 'concat')(embedding)
    # BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True))(word_embedding_dropout)
    BiLSTM = BatchNormalization(axis=1)(BiLSTM)
    BiLSTM_dropout = Dropout(0.5)(BiLSTM)

    mlp1_hidden1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(BiLSTM_dropout)
    mlp1_hidden1 = Dropout(0.5)(mlp1_hidden1)

    mlp1_hidden2 = TimeDistributed(Dense(100, activation='tanh'))(mlp1_hidden1)
    # output1 = TimeDistributed(Dense(5+1, activation='softmax'), name='BIOES')(decodelayer1)
    mlp1_hidden3 = TimeDistributed(Dense(5 + 1, activation=None))(mlp1_hidden2)
    crflayer1 = CRF(5 + 1, sparse_target=False, learn_mode='marginal', name='BIOES')
    output1 = crflayer1(mlp1_hidden3)

    mlp2_hidden1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(BiLSTM_dropout)
    mlp2_hidden1 = Dropout(0.5)(mlp2_hidden1)

    BiLSTM_pos = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(pos_embeding)
    BiLSTM_pos = Dropout(0.5)(BiLSTM_pos)
    mlp2_hidden1 = concatenate([BiLSTM_pos, mlp2_hidden1], axis=-1)

    mlp2_hidden2 = TimeDistributed(Dense(100, activation='tanh'))(mlp2_hidden1)
    mlp2_hidden3 = concatenate([mlp1_hidden2, mlp2_hidden2], axis=-1)
    mlp2_hidden3 = TimeDistributed(Dense(100, activation='tanh'))(mlp2_hidden3)
    # output2 = TimeDistributed(Dense(5+1, activation='softmax'), name='Type')(decodelayer2)
    mlp2_hidden3 = TimeDistributed(Dense(targetvocabsize + 1, activation=None))(mlp2_hidden3)
    crflayer2 = CRF(targetvocabsize + 1, sparse_target=False, name='Type', learn_mode='marginal')
    output2 = crflayer2(mlp2_hidden3)


    Models = Model([word_input, char_input], [output1, output2])
    # Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
    #                loss={'finall': crflayer.loss_function, 'BIOES': 'categorical_crossentropy', 'Type': 'categorical_crossentropy'},
    #                loss_weights={'finall': 1., 'BIOES': 1., 'Type': 1.},
    #                metrics={'finall': [crflayer.accuracy], 'BIOES': ['acc'], 'Type': ['acc']})

    Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
                   loss={'BIOES': crflayer1.loss_function, 'Type': crflayer2.loss_function},
                   loss_weights={'BIOES': 1., 'Type': 1.},
                   metrics={'BIOES': [crflayer2.accuracy], 'Type': [crflayer2.accuracy]})

    return Models