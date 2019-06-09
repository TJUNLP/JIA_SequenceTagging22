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




def BiLSTM_CRF_multi3_with2t_1(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim,
                              sourcecharsize, character_W, input_word_length, char_emd_dim,
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

    embedding = concatenate([word_embedding_dropout, char_macpool], axis=-1)

    # BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True,), merge_mode = 'concat')(embedding)
    # # BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True))(word_embedding_dropout)
    # BiLSTM = BatchNormalization(axis=1)(BiLSTM)
    # BiLSTM_dropout = Dropout(0.5)(BiLSTM)

    private_type_lstm_1 = Bidirectional(LSTM(128, return_sequences=True), merge_mode='concat')(embedding)
    private_type_lstm_1 = Dropout(0.5)(private_type_lstm_1)

    private_bioes_lstm_1 = Bidirectional(LSTM(128, return_sequences=True), merge_mode='concat')(embedding)
    private_bioes_lstm_1 = Dropout(0.5)(private_bioes_lstm_1)

    public_embedding = concatenate([private_type_lstm_1, private_bioes_lstm_1], axis=-1)
    public_lstm = Bidirectional(LSTM(128, return_sequences=True), merge_mode='concat')(public_embedding)
    public_lstm = Dropout(0.5)(public_lstm)

    public_hidden_1 = TimeDistributed(Dense(64, activation='tanh'))(public_lstm)
    output1 = TimeDistributed(Dense(2 + 1, activation='softmax'), name='PUBLIC')(public_hidden_1)

    private_type_lstm_2 = Bidirectional(LSTM(128, return_sequences=True), merge_mode='concat')(private_type_lstm_1)
    private_type_lstm_2 = Dropout(0.5)(private_type_lstm_2)

    private_type_embedding = concatenate([private_type_lstm_2, public_lstm], axis=-1)
    private_type_hidden1 = TimeDistributed(Dense(5 + 1, activation=None))(private_type_embedding)
    crflayer1 = CRF(5 + 1, sparse_target=False, learn_mode='marginal', name='TYPE')
    output2 = crflayer1(private_type_hidden1)

    private_bioes_lstm_2 = Bidirectional(LSTM(128, return_sequences=True), merge_mode='concat')(private_bioes_lstm_1)
    private_bioes_lstm_2 = Dropout(0.5)(private_bioes_lstm_2)

    private_bioes_embedding = concatenate([private_bioes_lstm_2, public_lstm], axis=-1)
    private_bioes_hidden_1 = TimeDistributed(Dense(5 + 1, activation=None))(private_bioes_embedding)
    crflayer2 = CRF(5 + 1, sparse_target=False, name='BIOES', learn_mode='marginal')
    output3 = crflayer2(private_bioes_hidden_1)

    Models = Model([word_input, char_input], [output1, output2, output3])
    # Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
    #                loss={'finall': crflayer.loss_function, 'BIOES': 'categorical_crossentropy', 'Type': 'categorical_crossentropy'},
    #                loss_weights={'finall': 1., 'BIOES': 1., 'Type': 1.},
    #                metrics={'finall': [crflayer.accuracy], 'BIOES': ['acc'], 'Type': ['acc']})

    Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
                   loss={'PUBLIC': 'categorical_crossentropy',
                         'TYPE': crflayer1.loss_function,
                         'BIOES': crflayer2.loss_function},
                   loss_weights={'BIOES': 1., 'TYPE': 1., 'PUBLIC': 0.5},
                   metrics={'PUBLIC': ['acc'], 'BIOES': [crflayer2.accuracy], 'TYPE': [crflayer2.accuracy]})

    return Models


def Model_BiLSTM_CRF_multi3_1(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim,
                              sourcecharsize, character_W, input_word_length, char_emd_dim,targetpossize,
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

    embedding = concatenate([word_embedding_dropout, char_macpool], axis=-1)

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True,), merge_mode = 'concat')(embedding)
    # BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True))(word_embedding_dropout)
    BiLSTM = BatchNormalization(axis=1)(BiLSTM)
    BiLSTM_dropout = Dropout(0.5)(BiLSTM)

    mlp3_hidden1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(BiLSTM_dropout)
    mlp3_hidden1 = Dropout(0.5)(mlp3_hidden1)
    mlp3_hidden2 = TimeDistributed(Dense(100, activation='tanh'))(mlp3_hidden1)
    output3 = TimeDistributed(Dense(targetpossize + 1, activation='softmax'), name='pos')(mlp3_hidden2)

    embedding2 = concatenate([BiLSTM_dropout, mlp3_hidden1], axis=-1)

    mlp1_hidden1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(embedding2)
    mlp1_hidden1 = Dropout(0.5)(mlp1_hidden1)
    mlp1_hidden2 = TimeDistributed(Dense(100, activation='tanh'))(mlp1_hidden1)
    # output1 = TimeDistributed(Dense(5+1, activation='softmax'), name='BIOES')(decodelayer1)
    mlp1_hidden3 = TimeDistributed(Dense(5 + 1, activation=None))(mlp1_hidden2)
    crflayer1 = CRF(5 + 1, sparse_target=False, learn_mode='marginal', name='BIOES')
    output1 = crflayer1(mlp1_hidden3)

    mlp2_hidden1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(embedding2)
    mlp2_hidden1 = Dropout(0.5)(mlp2_hidden1)
    mlp2_hidden2 = TimeDistributed(Dense(100, activation='tanh'))(mlp2_hidden1)
    # output2 = TimeDistributed(Dense(5+1, activation='softmax'), name='Type')(decodelayer2)
    mlp2_hidden3 = TimeDistributed(Dense(5 + 1, activation=None))(mlp2_hidden2)
    mlp2_hidden4 = average([mlp1_hidden3, mlp2_hidden3])
    crflayer2 = CRF(5 + 1, sparse_target=False, name='Type', learn_mode='marginal')
    output2 = crflayer2(mlp2_hidden4)

    Models = Model([word_input, char_input], [output1, output2, output3])
    # Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
    #                loss={'finall': crflayer.loss_function, 'BIOES': 'categorical_crossentropy', 'Type': 'categorical_crossentropy'},
    #                loss_weights={'finall': 1., 'BIOES': 1., 'Type': 1.},
    #                metrics={'finall': [crflayer.accuracy], 'BIOES': ['acc'], 'Type': ['acc']})

    Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
                   loss={'BIOES': crflayer1.loss_function,
                         'Type': crflayer2.loss_function, 'pos': 'categorical_crossentropy'},

    loss_weights={'BIOES': 1., 'Type': 1.,'pos': 0.5},
                   metrics={'BIOES': [crflayer2.accuracy], 'Type': [crflayer2.accuracy], 'pos': ['acc']})

    return Models

