# -*- encoding:utf-8 -*-

from keras.layers import Flatten,Lambda,Conv2D
from keras.layers.core import Dropout, Activation, Permute, RepeatVector
from keras.layers.merge import concatenate, multiply, Dot, average, add
from keras.layers import TimeDistributed, Input, Bidirectional, Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D, RepeatVector, AveragePooling1D
from keras.models import Model
from keras_contrib.layers import CRF
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



def BiLSTM_CRF_multi2_order3_Dense(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
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

    input_chunk = TimeDistributed(Dense(100, activation=None))(output1)
    embedding2 = concatenate([BiLSTM_dropout, input_chunk], axis=-1)

    mlp2_hidden1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(embedding2)
    mlp2_hidden1 = Dropout(0.5)(mlp2_hidden1)
    mlp2_hidden2 = TimeDistributed(Dense(100, activation='tanh'))(mlp2_hidden1)
    mlp2_hidden3 = TimeDistributed(Dense(100, activation='tanh'))(mlp2_hidden2)
    # output2 = TimeDistributed(Dense(5+1, activation='softmax'), name='Type')(decodelayer2)
    mlp2_hidden3 = TimeDistributed(Dense(5 + 1, activation=None))(mlp2_hidden3)
    # mlp2_hidden4 = average([mlp1_hidden3, mlp2_hidden3])
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


def BiLSTM_CRF_multi2_order3_DenseAvg(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
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

    input_chunk = TimeDistributed(Dense(100, activation=None))(output1)

    # input_chunk = Dropout(0)(input_chunk)
    # !!!!!!!!!!!!!!
    input_chunk = Dropout(0.5)(input_chunk)

    embedding2 = concatenate([BiLSTM_dropout, input_chunk], axis=-1)

    mlp2_hidden1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(embedding2)
    mlp2_hidden1 = Dropout(0.5)(mlp2_hidden1)
    mlp2_hidden2 = TimeDistributed(Dense(100, activation='tanh'))(mlp2_hidden1)
    mlp2_hidden3 = TimeDistributed(Dense(100, activation='tanh'))(mlp2_hidden2)
    # output2 = TimeDistributed(Dense(5+1, activation='softmax'), name='Type')(decodelayer2)
    mlp2_hidden3 = TimeDistributed(Dense(5 + 1, activation=None))(mlp2_hidden3)
    mlp2_hidden4 = average([mlp1_hidden3, mlp2_hidden3])
    crflayer2 = CRF(targetvocabsize + 1, sparse_target=False, name='Type', learn_mode='marginal')
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


def BiLSTM_CRF_multi2_order4_DenseAvg(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
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

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True,), merge_mode = 'concat')(embedding)
    # BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True))(word_embedding_dropout)
    BiLSTM = BatchNormalization(axis=1)(BiLSTM)
    BiLSTM_dropout = Dropout(0.5)(BiLSTM)

    mlp1_hidden1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(BiLSTM_dropout)
    mlp1_hidden1 = Dropout(0.5)(mlp1_hidden1)
    mlp1_hidden2 = TimeDistributed(Dense(100, activation='tanh'))(mlp1_hidden1)
    # output1 = TimeDistributed(Dense(5+1, activation='softmax'), name='BIOES')(decodelayer1)
    mlp1_hidden3 = TimeDistributed(Dense(5 + 1, activation=None))(mlp1_hidden2)
    crflayer1 = CRF(5 + 1, sparse_target=False, learn_mode='marginal')
    output1 = crflayer1(mlp1_hidden3)

    # input_chunk = TimeDistributed(Dense(100, activation=None))(output1)
    # !!!!!!!!!!!!!!
    input_chunk = Bidirectional(LSTM(50, return_sequences=True), merge_mode='concat')(output1)


    input_chunk = Dropout(0.5)(input_chunk)

    embedding2 = concatenate([BiLSTM_dropout, input_chunk], axis=-1)

    mlp2_hidden1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(embedding2)
    mlp2_hidden1 = Dropout(0.5)(mlp2_hidden1)
    mlp2_hidden2 = TimeDistributed(Dense(100, activation='tanh'))(mlp2_hidden1)
    mlp2_hidden3 = TimeDistributed(Dense(100, activation='tanh'))(mlp2_hidden2)
    # output2 = TimeDistributed(Dense(5+1, activation='softmax'), name='Type')(decodelayer2)
    mlp2_hidden3 = TimeDistributed(Dense(5 + 1, activation=None))(mlp2_hidden3)

    output2 = crflayer1(mlp2_hidden3)


    Models = Model([word_input, char_input], [output1, output2])
    # Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
    #                loss={'finall': crflayer.loss_function, 'BIOES': 'categorical_crossentropy', 'Type': 'categorical_crossentropy'},
    #                loss_weights={'finall': 1., 'BIOES': 1., 'Type': 1.},
    #                metrics={'finall': [crflayer.accuracy], 'BIOES': ['acc'], 'Type': ['acc']})

    Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
                   loss=[crflayer1.loss_function, crflayer1.loss_function],
                   loss_weights=[1., 1.],
                   metrics=[crflayer1.accuracy])

    return Models


def BiLSTM_CRF_multi2_order5_DenseAvg(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
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

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True,), merge_mode = 'concat')(embedding)
    # BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True))(word_embedding_dropout)
    BiLSTM = BatchNormalization(axis=1)(BiLSTM)
    BiLSTM_dropout = Dropout(0.5)(BiLSTM)

    # output1 = TimeDistributed(Dense(5+1, activation='softmax'), name='BIOES')(decodelayer1)
    mlp1_hidden3 = TimeDistributed(Dense(5 + 1, activation=None))(BiLSTM_dropout)
    crflayer1 = CRF(5 + 1, sparse_target=False, learn_mode='marginal')
    output1 = crflayer1(mlp1_hidden3)

    input_chunk = TimeDistributed(Dense(100, activation=None))(output1)

    # input_chunk = Dropout(0)(input_chunk)
    # !!!!!!!!!!!!!!
    input_chunk = Dropout(0.5)(input_chunk)

    embedding2 = concatenate([BiLSTM_dropout, input_chunk], axis=-1)

    mlp2_hidden1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(embedding2)
    mlp2_hidden2 = Dropout(0.5)(mlp2_hidden1)

    # output2 = TimeDistributed(Dense(5+1, activation='softmax'), name='Type')(decodelayer2)
    mlp2_hidden3 = TimeDistributed(Dense(5 + 1, activation=None))(mlp2_hidden2)

    output2 = crflayer1(mlp2_hidden3)


    Models = Model([word_input, char_input], [output1, output2])
    # Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
    #                loss={'finall': crflayer.loss_function, 'BIOES': 'categorical_crossentropy', 'Type': 'categorical_crossentropy'},
    #                loss_weights={'finall': 1., 'BIOES': 1., 'Type': 1.},
    #                metrics={'finall': [crflayer.accuracy], 'BIOES': ['acc'], 'Type': ['acc']})

    Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
                   loss=[crflayer1.loss_function, crflayer1.loss_function],
                   loss_weights=[1., 1.],
                   metrics=[crflayer1.accuracy])

    return Models

def BiLSTM_CRF_multi2_order6_Double(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
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

    output1 = TimeDistributed(Dense(5+1, activation='softmax'), name='BIOES')(BiLSTM_dropout)

    hiddenlayer2_1 = TimeDistributed(Dense(targetvocabsize+1))(BiLSTM_dropout)
    crflayer = CRF(targetvocabsize+1, sparse_target=False, name='Type')
    output2 = crflayer(hiddenlayer2_1)


    Models = Model([word_input, char_input], [output1, output2])

    # Models.compile(loss=crflayer.loss_function, optimizer=optimizers.RMSprop(lr=0.001), metrics=[crflayer.accuracy])
    Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
                   loss={'Type': crflayer.loss_function, 'BIOES': 'categorical_crossentropy'},
                   loss_weights={'BIOES': 1., 'Type': 1.},
                   metrics={'BIOES': ['acc'], 'Type': [crflayer.accuracy]})
    return Models


def BiLSTM_CRF_multi2_order7_Serial(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
                                    output_seq_lenth,
                                    hidden_dim, emd_dim,
                                    sourcecharsize, character_W, input_word_length, char_emd_dim,
                                    sourcepossize, pos_W, pos_emd_dim, batch_size=32,
                                    loss='categorical_crossentropy', optimizer='rmsprop'):

    word_input = Input(shape=(input_seq_lenth,), dtype='int32')
    char_input = Input(shape=(input_seq_lenth, input_word_length,), dtype='int32')

    char_embedding = Embedding(input_dim=sourcecharsize,
                               output_dim=char_emd_dim,
                               batch_input_shape=(batch_size, input_seq_lenth, input_word_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[character_W])

    char_embedding2 = TimeDistributed(char_embedding)(char_input)

    char_cnn = TimeDistributed(Conv1D(50, 3, activation='relu', padding='same'))(char_embedding2)

    char_macpool = TimeDistributed(GlobalMaxPooling1D())(char_cnn)

    char_macpool = Dropout(0.25)(char_macpool)

    word_embedding = Embedding(input_dim=sourcevocabsize + 1,
                               output_dim=emd_dim,
                               input_length=input_seq_lenth,
                               mask_zero=False,
                               trainable=True,
                               weights=[source_W])(word_input)

    word_embedding_dropout = Dropout(0.5)(word_embedding)

    embedding1 = concatenate([word_embedding_dropout, char_macpool], axis=-1)

    BiLSTM1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(embedding1)
    BiLSTM1 = BatchNormalization(axis=1)(BiLSTM1)
    BiLSTM1_dropout = Dropout(0.5)(BiLSTM1)

    input_chunk = TimeDistributed(Dense(100, activation='tanh'))(BiLSTM1_dropout)
    input_chunk = Dropout(0.5)(input_chunk)
    output1 = TimeDistributed(Dense(5 + 1, activation='softmax'), name='BIOES')(input_chunk)


    embedding2 = concatenate([word_embedding_dropout, char_macpool, input_chunk], axis=-1)

    BiLSTM2 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(embedding2)
    BiLSTM2 = BatchNormalization(axis=1)(BiLSTM2)
    BiLSTM2_dropout = Dropout(0.5)(BiLSTM2)

    hiddenlayer2_1 = TimeDistributed(Dense(5 + 1))(BiLSTM2_dropout)
    crflayer = CRF(5 + 1, sparse_target=False, name='Type')
    output2 = crflayer(hiddenlayer2_1)

    Models = Model([word_input, char_input], [output1, output2])

    # Models.compile(loss=crflayer.loss_function, optimizer=optimizers.RMSprop(lr=0.001), metrics=[crflayer.accuracy])
    Models.compile(optimizer=optimizers.Adam(lr=0.001),
                   loss={'Type': crflayer.loss_function, 'BIOES': 'categorical_crossentropy'},
                   loss_weights={'BIOES': 1., 'Type': 1.},
                   metrics={'BIOES': ['acc'], 'Type': [crflayer.accuracy]})
    return Models


def BiLSTM_CRF_multi2_order7_Serial2(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
                                    output_seq_lenth,
                                    hidden_dim, emd_dim,
                                    sourcecharsize, character_W, input_word_length, char_emd_dim,
                                    sourcepossize, pos_W, pos_emd_dim, batch_size=32,
                                    loss='categorical_crossentropy', optimizer='rmsprop'):

    word_input = Input(shape=(input_seq_lenth,), dtype='int32')
    char_input = Input(shape=(input_seq_lenth, input_word_length,), dtype='int32')

    char_embedding = Embedding(input_dim=sourcecharsize,
                               output_dim=char_emd_dim,
                               batch_input_shape=(batch_size, input_seq_lenth, input_word_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[character_W])

    char_embedding2 = TimeDistributed(char_embedding)(char_input)

    char_cnn = TimeDistributed(Conv1D(50, 3, activation='relu', padding='same'))(char_embedding2)

    char_macpool = TimeDistributed(GlobalMaxPooling1D())(char_cnn)

    char_macpool = Dropout(0.25)(char_macpool)

    word_embedding = Embedding(input_dim=sourcevocabsize + 1,
                               output_dim=emd_dim,
                               input_length=input_seq_lenth,
                               mask_zero=False,
                               trainable=True,
                               weights=[source_W])(word_input)

    word_embedding_dropout = Dropout(0.5)(word_embedding)

    embedding1 = concatenate([word_embedding_dropout, char_macpool], axis=-1)

    BiLSTM1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(embedding1)
    BiLSTM1 = BatchNormalization(axis=1)(BiLSTM1)
    BiLSTM1_dropout = Dropout(0.5)(BiLSTM1)

    output1 = TimeDistributed(Dense(5 + 1, activation='softmax'), name='BIOES')(BiLSTM1_dropout)


    BiLSTM2 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(embedding1)
    BiLSTM2 = BatchNormalization(axis=1)(BiLSTM2)
    BiLSTM2_dropout = Dropout(0.5)(BiLSTM2)

    input_chunk = TimeDistributed(Dense(50, activation=None))(output1)
    input_chunk = Dropout(0.25)(input_chunk)

    embedding2 = concatenate([BiLSTM2_dropout, input_chunk], axis=-1)

    output_cnn = Conv1D(100, 3, activation='relu', padding='same')(embedding2)

    hiddenlayer2_1 = TimeDistributed(Dense(5 + 1))(output_cnn)
    crflayer = CRF(5 + 1, sparse_target=False, name='Type')
    output2 = crflayer(hiddenlayer2_1)

    Models = Model([word_input, char_input], [output1, output2])

    # Models.compile(loss=crflayer.loss_function, optimizer=optimizers.RMSprop(lr=0.001), metrics=[crflayer.accuracy])
    Models.compile(optimizer=optimizers.Adam(lr=0.001),
                   loss={'Type': crflayer.loss_function, 'BIOES': 'categorical_crossentropy'},
                   loss_weights={'BIOES': 1., 'Type': 1.},
                   metrics={'BIOES': ['acc'], 'Type': [crflayer.accuracy]})
    return Models


def BiLSTM_CRF_multi2_order7_Serial3(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
                                    output_seq_lenth,
                                    hidden_dim, emd_dim,
                                    sourcecharsize, character_W, input_word_length, char_emd_dim,
                                    sourcepossize, pos_W, pos_emd_dim, batch_size=32,
                                    loss='categorical_crossentropy', optimizer='rmsprop'):

    word_input = Input(shape=(input_seq_lenth,), dtype='int32')
    char_input = Input(shape=(input_seq_lenth, input_word_length,), dtype='int32')

    char_embedding = Embedding(input_dim=sourcecharsize,
                               output_dim=char_emd_dim,
                               batch_input_shape=(batch_size, input_seq_lenth, input_word_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[character_W])

    char_embedding2 = TimeDistributed(char_embedding)(char_input)

    char_cnn = TimeDistributed(Conv1D(50, 3, activation='relu', padding='same'))(char_embedding2)

    char_macpool = TimeDistributed(GlobalMaxPooling1D())(char_cnn)

    char_macpool = Dropout(0.25)(char_macpool)

    word_embedding = Embedding(input_dim=sourcevocabsize + 1,
                               output_dim=emd_dim,
                               input_length=input_seq_lenth,
                               mask_zero=False,
                               trainable=True,
                               weights=[source_W])(word_input)

    word_embedding_dropout = Dropout(0.5)(word_embedding)

    embedding1 = concatenate([word_embedding_dropout, char_macpool], axis=-1)

    BiLSTM1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(embedding1)
    BiLSTM1 = BatchNormalization(axis=1)(BiLSTM1)
    BiLSTM1_dropout = Dropout(0.5)(BiLSTM1)

    output1 = TimeDistributed(Dense(5 + 1, activation='softmax'), name='BIOES')(BiLSTM1_dropout)


    BiLSTM2 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(embedding1)
    BiLSTM2 = BatchNormalization(axis=1)(BiLSTM2)
    BiLSTM2_dropout = Dropout(0.5)(BiLSTM2)

    input_chunk = TimeDistributed(Dense(50, activation=None))(output1)
    input_chunk = Dropout(0.25)(input_chunk)

    embedding2 = concatenate([BiLSTM2_dropout, input_chunk], axis=-1)

    output_cnn = Conv1D(100, 3, activation='relu', padding='same')(embedding2)

    hiddenlayer2_1 = TimeDistributed(Dense(5 + 1))(output_cnn)
    crflayer = CRF(5 + 1, sparse_target=False, name='Type')
    output2 = crflayer(hiddenlayer2_1)

    Models = Model([word_input, char_input], [output1, output2])

    # Models.compile(loss=crflayer.loss_function, optimizer=optimizers.RMSprop(lr=0.001), metrics=[crflayer.accuracy])
    Models.compile(optimizer=optimizers.Adam(lr=0.001),
                   loss={'Type': crflayer.loss_function, 'BIOES': 'categorical_crossentropy'},
                   loss_weights={'BIOES': 1., 'Type': 1.},
                   metrics={'BIOES': ['acc'], 'Type': [crflayer.accuracy]})
    return Models



def BiLSTM_CRF_multi2_order7_Serial_All(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
                                    output_seq_lenth,
                                    hidden_dim, emd_dim,
                                    sourcecharsize, character_W, input_word_length, char_emd_dim,
                                    sourcepossize, pos_W, pos_emd_dim, batch_size=32,
                                    loss='categorical_crossentropy', optimizer='rmsprop'):

    word_input = Input(shape=(input_seq_lenth,), dtype='int32')
    char_input = Input(shape=(input_seq_lenth, input_word_length,), dtype='int32')

    char_embedding = Embedding(input_dim=sourcecharsize,
                               output_dim=char_emd_dim,
                               batch_input_shape=(batch_size, input_seq_lenth, input_word_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[character_W])

    char_embedding2 = TimeDistributed(char_embedding)(char_input)

    char_cnn = TimeDistributed(Conv1D(50, 3, activation='relu', padding='same'))(char_embedding2)

    char_macpool = TimeDistributed(GlobalMaxPooling1D())(char_cnn)

    char_macpool = Dropout(0.25)(char_macpool)

    word_embedding = Embedding(input_dim=sourcevocabsize + 1,
                               output_dim=emd_dim,
                               input_length=input_seq_lenth,
                               mask_zero=False,
                               trainable=True,
                               weights=[source_W])(word_input)

    word_embedding_dropout = Dropout(0.5)(word_embedding)

    embedding1 = concatenate([word_embedding_dropout, char_macpool], axis=-1)

    BiLSTM1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(embedding1)
    BiLSTM1 = BatchNormalization(axis=1)(BiLSTM1)
    BiLSTM1_dropout = Dropout(0.5)(BiLSTM1)

    output1 = TimeDistributed(Dense(5 + 1, activation='softmax'), name='BIOES')(BiLSTM1_dropout)

    input_chunk = TimeDistributed(Dense(50, activation=None))(output1)
    input_chunk = Dropout(0.25)(input_chunk)

    embedding2 = concatenate([word_embedding_dropout, char_macpool, input_chunk], axis=-1)

    BiLSTM2 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(embedding2)
    BiLSTM2 = BatchNormalization(axis=1)(BiLSTM2)
    BiLSTM2_dropout = Dropout(0.5)(BiLSTM2)

    hiddenlayer2_1 = TimeDistributed(Dense(targetvocabsize + 1))(BiLSTM2_dropout)
    crflayer = CRF(targetvocabsize + 1, sparse_target=False, name='Type')
    output2 = crflayer(hiddenlayer2_1)

    Models = Model([word_input, char_input], [output1, output2])

    # Models.compile(loss=crflayer.loss_function, optimizer=optimizers.RMSprop(lr=0.001), metrics=[crflayer.accuracy])
    Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
                   loss={'Type': crflayer.loss_function, 'BIOES': 'categorical_crossentropy'},
                   loss_weights={'BIOES': 1., 'Type': 1.},
                   metrics={'BIOES': ['acc'], 'Type': [crflayer.accuracy]})
    return Models



def BiLSTM_CRF_multi2_Attention(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
                                    output_seq_lenth,
                                    hidden_dim, emd_dim,
                                    sourcecharsize, character_W, input_word_length, char_emd_dim,
                                    sourcepossize, pos_W, pos_emd_dim, batch_size=32,
                                    loss='categorical_crossentropy', optimizer='rmsprop'):

    word_input = Input(shape=(input_seq_lenth,), dtype='int32')
    char_input = Input(shape=(input_seq_lenth, input_word_length,), dtype='int32')

    char_embedding = Embedding(input_dim=sourcecharsize,
                               output_dim=char_emd_dim,
                               batch_input_shape=(batch_size, input_seq_lenth, input_word_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[character_W])

    char_embedding2 = TimeDistributed(char_embedding)(char_input)

    char_cnn = TimeDistributed(Conv1D(50, 3, activation='relu', padding='same'))(char_embedding2)

    char_macpool = TimeDistributed(GlobalMaxPooling1D())(char_cnn)

    char_macpool = Dropout(0.25)(char_macpool)

    word_embedding = Embedding(input_dim=sourcevocabsize + 1,
                               output_dim=emd_dim,
                               input_length=input_seq_lenth,
                               mask_zero=False,
                               trainable=True,
                               weights=[source_W])(word_input)

    word_embedding_dropout = Dropout(0.5)(word_embedding)

    embedding1 = concatenate([word_embedding_dropout, char_macpool], axis=-1)

    BiLSTM1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(embedding1)
    BiLSTM1 = BatchNormalization(axis=1)(BiLSTM1)
    BiLSTM1_dropout = Dropout(0.5)(BiLSTM1)

    output_pre1 = TimeDistributed(Dense(hidden_dim, activation='relu'))(BiLSTM1_dropout)
    output1 = TimeDistributed(Dense(1, activation='sigmoid'), name='BIOES')(output_pre1)

    Attention = TimeDistributed(RepeatVector(hidden_dim * 2))(output1)
    Attention = TimeDistributed(Flatten())(Attention)
    Attention = multiply([BiLSTM1, Attention])
    Attention = BatchNormalization(axis=1)(Attention)
    Attention_dropout = Dropout(0.5)(Attention)

    BiLSTM2 = Bidirectional(LSTM(hidden_dim // 2, return_sequences=True), merge_mode='concat')(Attention_dropout)
    BiLSTM2 = BatchNormalization(axis=1)(BiLSTM2)
    BiLSTM2_dropout = Dropout(0.5)(BiLSTM2)

    hiddenlayer2_1 = TimeDistributed(Dense(targetvocabsize + 1))(BiLSTM2_dropout)
    crflayer = CRF(targetvocabsize + 1, sparse_target=False, name='Type')
    output2 = crflayer(hiddenlayer2_1)

    Models = Model([word_input, char_input], [output1, output2])

    # Models.compile(loss=crflayer.loss_function, optimizer=optimizers.RMSprop(lr=0.001), metrics=[crflayer.accuracy])
    Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
                   loss={'Type': crflayer.loss_function, 'BIOES': 'mean_squared_error'},
                   loss_weights={'BIOES': 1., 'Type': 1.},
                   metrics={'BIOES': ['acc'], 'Type': [crflayer.accuracy]})
    return Models


def BiLSTM_CRF_multi2_order7_Serial_All_2(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
                                    output_seq_lenth,
                                    hidden_dim, emd_dim,
                                    sourcecharsize, character_W, input_word_length, char_emd_dim,
                                    sourcepossize, pos_W, pos_emd_dim, batch_size=32,
                                    loss='categorical_crossentropy', optimizer='rmsprop'):

    word_input = Input(shape=(input_seq_lenth,), dtype='int32')
    char_input = Input(shape=(input_seq_lenth, input_word_length,), dtype='int32')

    char_embedding = Embedding(input_dim=sourcecharsize,
                               output_dim=char_emd_dim,
                               batch_input_shape=(batch_size, input_seq_lenth, input_word_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[character_W])

    char_embedding2 = TimeDistributed(char_embedding)(char_input)

    char_cnn = TimeDistributed(Conv1D(50, 3, activation='relu', padding='same'))(char_embedding2)

    char_macpool = TimeDistributed(GlobalMaxPooling1D())(char_cnn)

    char_macpool = Dropout(0.25)(char_macpool)

    word_embedding = Embedding(input_dim=sourcevocabsize + 1,
                               output_dim=emd_dim,
                               input_length=input_seq_lenth,
                               mask_zero=False,
                               trainable=True,
                               weights=[source_W])(word_input)

    word_embedding_dropout = Dropout(0.5)(word_embedding)

    embedding1 = concatenate([word_embedding_dropout, char_macpool], axis=-1)

    BiLSTM1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(embedding1)
    BiLSTM1 = BatchNormalization(axis=1)(BiLSTM1)
    BiLSTM1_dropout = Dropout(0.5)(BiLSTM1)

    output1 = TimeDistributed(Dense(5 + 1, activation='softmax'), name='BIOES')(BiLSTM1_dropout)

    input_chunk = TimeDistributed(Dense(50, activation=None))(output1)
    input_chunk = Dropout(0.25)(input_chunk)

    embedding2 = concatenate([word_embedding_dropout, char_macpool, output1], axis=-1)

    BiLSTM2 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(embedding2)
    BiLSTM2 = BatchNormalization(axis=1)(BiLSTM2)
    BiLSTM2_dropout = Dropout(0.5)(BiLSTM2)

    hiddenlayer2_1 = TimeDistributed(Dense(targetvocabsize + 1))(BiLSTM2_dropout)
    crflayer = CRF(targetvocabsize + 1, sparse_target=False, name='Type')
    output2 = crflayer(hiddenlayer2_1)

    Models = Model([word_input, char_input], [output1, output2])

    # Models.compile(loss=crflayer.loss_function, optimizer=optimizers.RMSprop(lr=0.001), metrics=[crflayer.accuracy])
    Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
                   loss={'Type': crflayer.loss_function, 'BIOES': 'categorical_crossentropy'},
                   loss_weights={'BIOES': 1., 'Type': 1.},
                   metrics={'BIOES': ['acc'], 'Type': [crflayer.accuracy]})
    return Models


def BiLSTM_CRF_multi2_order7_Serial_Softmax(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
                                    output_seq_lenth,
                                    hidden_dim, emd_dim,
                                    sourcecharsize, character_W, input_word_length, char_emd_dim,
                                    sourcepossize, pos_W, pos_emd_dim, batch_size=32,
                                    loss='categorical_crossentropy', optimizer='rmsprop'):

    word_input = Input(shape=(input_seq_lenth,), dtype='int32')
    char_input = Input(shape=(input_seq_lenth, input_word_length,), dtype='int32')

    char_embedding = Embedding(input_dim=sourcecharsize,
                               output_dim=char_emd_dim,
                               batch_input_shape=(batch_size, input_seq_lenth, input_word_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[character_W])

    char_embedding2 = TimeDistributed(char_embedding)(char_input)

    char_cnn = TimeDistributed(Conv1D(50, 3, activation='relu', padding='same'))(char_embedding2)

    char_macpool = TimeDistributed(GlobalMaxPooling1D())(char_cnn)

    char_macpool = Dropout(0.25)(char_macpool)

    word_embedding = Embedding(input_dim=sourcevocabsize + 1,
                               output_dim=emd_dim,
                               input_length=input_seq_lenth,
                               mask_zero=False,
                               trainable=True,
                               weights=[source_W])(word_input)

    word_embedding_dropout = Dropout(0.5)(word_embedding)

    embedding1 = concatenate([word_embedding_dropout, char_macpool], axis=-1)

    BiLSTM1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(embedding1)
    BiLSTM1 = BatchNormalization(axis=1)(BiLSTM1)
    BiLSTM1_dropout = Dropout(0.5)(BiLSTM1)

    output1 = TimeDistributed(Dense(5 + 1, activation='softmax'), name='BIOES')(BiLSTM1_dropout)


    embedding2 = concatenate([word_embedding_dropout, char_macpool, BiLSTM1_dropout], axis=-1)

    BiLSTM2 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(embedding2)
    BiLSTM2 = BatchNormalization(axis=1)(BiLSTM2)
    BiLSTM2_dropout = Dropout(0.5)(BiLSTM2)

    output2 = TimeDistributed(Dense(5 + 1, activation='softmax'), name='Type')(BiLSTM2_dropout)

    Models = Model([word_input, char_input], [output1, output2])

    # Models.compile(loss=crflayer.loss_function, optimizer=optimizers.RMSprop(lr=0.001), metrics=[crflayer.accuracy])
    Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
                   loss={'Type': 'categorical_crossentropy', 'BIOES': 'categorical_crossentropy'},
                   loss_weights={'BIOES': 1., 'Type': 1.},
                   metrics={'BIOES': ['acc'], 'Type': ['acc']})
    return Models


def BiLSTM_CRF_multi2_order3_DenseAvg_crf_softmax(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
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

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True,), merge_mode = 'concat')(embedding)
    # BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True))(word_embedding_dropout)
    BiLSTM = BatchNormalization(axis=1)(BiLSTM)
    BiLSTM_dropout = Dropout(0.5)(BiLSTM)

    mlp1_hidden1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(BiLSTM_dropout)
    mlp1_hidden1 = Dropout(0.5)(mlp1_hidden1)
    # mlp1_hidden2 = TimeDistributed(Dense(100, activation='tanh'))(mlp1_hidden1)
    # output1 = TimeDistributed(Dense(5+1, activation='softmax'), name='BIOES')(decodelayer1)
    mlp1_hidden3 = TimeDistributed(Dense(5 + 1, activation=None))(mlp1_hidden1)
    crflayer1 = CRF(5 + 1, sparse_target=False, learn_mode='marginal', name='BIOES')
    output1 = crflayer1(mlp1_hidden3)

    # input_chunk = TimeDistributed(Dense(100, activation=None))(output1)
    input_chunk = Bidirectional(LSTM(100, return_sequences=True), merge_mode='concat')(output1)

    # input_chunk = Dropout(0)(input_chunk)
    # !!!!!!!!!!!!!!
    input_chunk = Dropout(0.5)(input_chunk)

    embedding2 = concatenate([BiLSTM_dropout, input_chunk], axis=-1)

    mlp2_hidden1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(embedding2)
    mlp2_hidden1 = Dropout(0.5)(mlp2_hidden1)
    # mlp2_hidden2 = TimeDistributed(Dense(100, activation='tanh'))(mlp2_hidden1)
    # mlp2_hidden3 = TimeDistributed(Dense(100, activation='tanh'))(mlp2_hidden2)
    # mlp2_hidden3 = TimeDistributed(Dense(5 + 1, activation=None))(mlp2_hidden3)
    # mlp2_hidden4 = average([mlp1_hidden3, mlp2_hidden3])

    output2 = TimeDistributed(Dense(5 + 1, activation='softmax'), name='Type')(mlp2_hidden1)

    Models = Model([word_input, char_input], [output1, output2])
    # Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
    #                loss={'finall': crflayer.loss_function, 'BIOES': 'categorical_crossentropy', 'Type': 'categorical_crossentropy'},
    #                loss_weights={'finall': 1., 'BIOES': 1., 'Type': 1.},
    #                metrics={'finall': [crflayer.accuracy], 'BIOES': ['acc'], 'Type': ['acc']})

    Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
                   loss={'BIOES': crflayer1.loss_function, 'Type': 'categorical_crossentropy'},
                   loss_weights={'BIOES': 1., 'Type': 1.},
                   metrics={'BIOES': [crflayer1.accuracy], 'Type': ['acc']})

    return Models


def BiLSTM_CRF_multi2_order3_DenseAvg_softmax_softmax(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
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

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True,), merge_mode = 'concat')(embedding)
    # BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True))(word_embedding_dropout)
    BiLSTM = BatchNormalization(axis=1)(BiLSTM)
    BiLSTM_dropout = Dropout(0.5)(BiLSTM)

    mlp1_hidden1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(BiLSTM_dropout)
    mlp1_hidden1 = Dropout(0.5)(mlp1_hidden1)
    mlp1_hidden2 = TimeDistributed(Dense(100, activation='tanh'))(mlp1_hidden1)

    # mlp1_hidden3 = TimeDistributed(Dense(5 + 1, activation=None))(mlp1_hidden2)
    # crflayer1 = CRF(5 + 1, sparse_target=False, learn_mode='marginal', name='BIOES')
    # output1 = crflayer1(mlp1_hidden3)

    mlp1_hidden3 = mlp1_hidden2
    output1 = TimeDistributed(Dense(5+1, activation='softmax'), name='BIOES')(mlp1_hidden3)


    input_chunk = TimeDistributed(Dense(100, activation=None))(output1)

    # input_chunk = Dropout(0)(input_chunk)
    # !!!!!!!!!!!!!!
    input_chunk = Dropout(0.5)(input_chunk)

    embedding2 = concatenate([BiLSTM_dropout, input_chunk], axis=-1)

    mlp2_hidden1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(embedding2)
    mlp2_hidden1 = Dropout(0.5)(mlp2_hidden1)
    mlp2_hidden2 = TimeDistributed(Dense(100, activation='tanh'))(mlp2_hidden1)
    mlp2_hidden3 = TimeDistributed(Dense(100, activation='tanh'))(mlp2_hidden2)

    mlp2_hidden4 = average([mlp1_hidden3, mlp2_hidden3])
    output2 = TimeDistributed(Dense(5+1, activation='softmax'), name='Type')(mlp2_hidden4)

    Models = Model([word_input, char_input], [output1, output2])

    # Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
    #                loss={'BIOES': crflayer1.loss_function, 'Type': crflayer2.loss_function},
    #                loss_weights={'BIOES': 1., 'Type': 1.},
    #                metrics={'BIOES': [crflayer2.accuracy], 'Type': [crflayer2.accuracy]})

    Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
                   loss={'BIOES': 'categorical_crossentropy', 'Type': 'categorical_crossentropy'},
                   loss_weights={'BIOES': 1., 'Type': 1.},
                   metrics={'BIOES': ['acc'], 'Type': ['acc']})

    return Models


def BiLSTM_CRF_multi2_order3_Coor(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
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

    embedding = concatenate([word_embedding_dropout, char_macpool], axis=-1)

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True,), merge_mode = 'concat')(embedding)
    # BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True))(word_embedding_dropout)
    BiLSTM = BatchNormalization(axis=1)(BiLSTM)
    BiLSTM_dropout = Dropout(0.5)(BiLSTM)

    mlp1_hidden1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(BiLSTM_dropout)
    mlp1_hidden1 = Dropout(0.5)(mlp1_hidden1)
    mlp1_hidden2 = TimeDistributed(Dense(100, activation='tanh'))(mlp1_hidden1)
    mlp0_hidden = TimeDistributed(Dense(100, activation='tanh'))
    mlp0_hidden1 = mlp0_hidden(mlp1_hidden1)
    mlp1_hidden3 = add([mlp0_hidden1, mlp1_hidden2])
    mlp1_hidden4 = TimeDistributed(Dense(5 + 1, activation=None))(mlp1_hidden3)
    crflayer1 = CRF(5 + 1, sparse_target=False, learn_mode='marginal', name='BIOES')
    output1 = crflayer1(mlp1_hidden4)

    input_chunk = TimeDistributed(Dense(100, activation=None))(output1)
    embedding2 = concatenate([BiLSTM_dropout, input_chunk], axis=-1)

    mlp2_hidden1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(embedding2)
    mlp2_hidden1 = Dropout(0.5)(mlp2_hidden1)
    mlp2_hidden2 = TimeDistributed(Dense(100, activation='tanh'))(mlp2_hidden1)
    mlp0_hidden2 = mlp0_hidden(mlp2_hidden1)
    mlp2_hidden3 = add([mlp0_hidden2, mlp2_hidden2])
    mlp2_hidden4 = TimeDistributed(Dense(5 + 1, activation=None))(mlp2_hidden3)
    # mlp2_hidden4 = average([mlp1_hidden3, mlp2_hidden3])
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


def BiLSTM_CRF_multi2_order3_full(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
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

    input_chunk2 = Flatten()(output1)
    input_chunk1 = Dense(100, activation=None)(input_chunk2)
    input_chunk = RepeatVector(input_seq_lenth)(input_chunk1)

    # input_chunk = TimeDistributed(Dense(100, activation=None))(output1)
    embedding2 = concatenate([BiLSTM_dropout, input_chunk], axis=-1)

    mlp2_hidden1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(embedding2)
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



def BiLSTM_CRF_multi2_order31(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
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
    mlp2_hidden2 = TimeDistributed(Dense(100, activation='tanh'))(mlp2_hidden2)
    # output2 = TimeDistributed(Dense(5+1, activation='softmax'), name='Type')(decodelayer2)
    mlp2_hidden3 = TimeDistributed(Dense(5 + 1, activation=None))(mlp2_hidden2)
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



def BiLSTM_CRF_multi2_order3_split(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
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

    BiLSTM_w = Bidirectional(LSTM(hidden_dim, return_sequences=True,), merge_mode = 'concat')(word_embedding_dropout)
    BiLSTM_c = Bidirectional(LSTM(hidden_dim, return_sequences=True,), merge_mode = 'concat')(char_macpool)

    BiLSTM_all = concatenate([BiLSTM_w, BiLSTM_c],axis=-1)
    BiLSTM_all = BatchNormalization(axis=1)(BiLSTM_all)
    BiLSTM_dropout = Dropout(0.5)(BiLSTM_all)



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


def BiLSTM_CRF_multi2_order_pos(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim,
                              sourcecharsize, character_W, input_word_length, char_emd_dim,
                              sourcepossize, pos_W,pos_emd_dim,
                            batch_size=32, pos_k=3,
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




    embedding = concatenate([word_embedding_dropout, char_macpool], axis=-1)

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True,), merge_mode = 'concat')(embedding)
    # BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True))(word_embedding_dropout)
    BiLSTM = BatchNormalization(axis=1)(BiLSTM)
    BiLSTM_dropout = Dropout(0.5)(BiLSTM)

    BiLSTM_pos = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(pos_macpool)
    BiLSTM_pos = Dropout(0.5)(BiLSTM_pos)
    BiLSTM_output = concatenate([BiLSTM_dropout, BiLSTM_pos], axis=-1)

    mlp1_hidden1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(BiLSTM_output)
    mlp1_hidden1 = Dropout(0.5)(mlp1_hidden1)
    mlp1_hidden2 = TimeDistributed(Dense(100, activation='tanh'))(mlp1_hidden1)
    # output1 = TimeDistributed(Dense(5+1, activation='softmax'), name='BIOES')(decodelayer1)
    mlp1_hidden3 = TimeDistributed(Dense(5 + 1, activation=None))(mlp1_hidden2)
    crflayer1 = CRF(5 + 1, sparse_target=False, learn_mode='marginal', name='BIOES')
    output1 = crflayer1(mlp1_hidden3)

    mlp2_hidden1 = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(BiLSTM_output)
    mlp2_hidden1 = Dropout(0.5)(mlp2_hidden1)

    # BiLSTM_pos = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='concat')(pos_macpool)
    # BiLSTM_pos = Dropout(0.5)(BiLSTM_pos)
    # mlp2_hidden1 = concatenate([BiLSTM_pos, mlp2_hidden1], axis=-1)

    mlp2_hidden2 = TimeDistributed(Dense(100, activation='tanh'))(mlp2_hidden1)
    mlp2_hidden2 = TimeDistributed(Dense(100, activation='tanh'))(mlp2_hidden2)
    # output2 = TimeDistributed(Dense(5+1, activation='softmax'), name='Type')(decodelayer2)
    mlp2_hidden3 = TimeDistributed(Dense(5 + 1, activation=None))(mlp2_hidden2)
    mlp2_hidden4 = average([mlp1_hidden3, mlp2_hidden3])
    crflayer2 = CRF(5 + 1, sparse_target=False, name='Type', learn_mode='marginal')
    output2 = crflayer2(mlp2_hidden4)


    Models = Model([word_input, char_input, pos_input], [output1, output2])
    # Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
    #                loss={'finall': crflayer.loss_function, 'BIOES': 'categorical_crossentropy', 'Type': 'categorical_crossentropy'},
    #                loss_weights={'finall': 1., 'BIOES': 1., 'Type': 1.},
    #                metrics={'finall': [crflayer.accuracy], 'BIOES': ['acc'], 'Type': ['acc']})

    Models.compile(optimizer=optimizers.RMSprop(lr=0.001),
                   loss={'BIOES': crflayer1.loss_function, 'Type': crflayer2.loss_function},
                   loss_weights={'BIOES': 1., 'Type': 1.},
                   metrics={'BIOES': [crflayer2.accuracy], 'Type': [crflayer2.accuracy]})

    return Models
