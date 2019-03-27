# coding:utf-8

from keras.layers.core import Dropout,RepeatVector, Reshape
from keras.layers.merge import concatenate, add, subtract, average, maximum
from keras.layers import TimeDistributed, Input, Bidirectional, Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D
from keras.models import Model
from keras import optimizers
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras import backend as K


# def Model_LSTM_BiLSTM_LSTM(wordvocabsize, targetvocabsize, charvobsize,
#                      word_W, char_W,
#                      input_fragment_lenth, input_leftcontext_lenth, input_rightcontext_lenth, input_maxword_length,
#                      w2v_k, c2v_k,
#                      hidden_dim=200, batch_size=32,
#                      optimizer='rmsprop'):
#
#     word_input_fragment = Input(shape=(input_fragment_lenth,), dtype='int32')
#     word_embedding_fragment = Embedding(input_dim=wordvocabsize + 1,
#                                output_dim=w2v_k,
#                                input_length=input_fragment_lenth,
#                                mask_zero=True,
#                                trainable=True,
#                                weights=[word_W])(word_input_fragment)
#     word_embedding_fragment = Dropout(0.5)(word_embedding_fragment)
#
#     char_input_fragment = Input(shape=(input_fragment_lenth, input_maxword_length,), dtype='int32')
#     char_embedding_fragment = TimeDistributed(Embedding(input_dim=charvobsize,
#                                output_dim=c2v_k,
#                                batch_input_shape=(batch_size, input_fragment_lenth, input_maxword_length),
#                                mask_zero=False,
#                                trainable=True,
#                                weights=[char_W]))(char_input_fragment)
#
#     char_cnn_fragment = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))
#     char_embedding_fragment = char_cnn_fragment(char_embedding_fragment)
#     char_embedding_fragment = TimeDistributed(GlobalMaxPooling1D())(char_embedding_fragment)
#     char_embedding_fragment = Dropout(0.25)(char_embedding_fragment)
#
#
#     word_input_leftcontext = Input(shape=(input_leftcontext_lenth,), dtype='int32')
#     word_embedding_leftcontext = Embedding(input_dim=wordvocabsize + 1,
#                                output_dim=w2v_k,
#                                input_length=input_leftcontext_lenth,
#                                mask_zero=True,
#                                trainable=True,
#                                weights=[word_W])(word_input_leftcontext)
#     word_embedding_leftcontext = Dropout(0.5)(word_embedding_leftcontext)
#
#     char_input_leftcontext = Input(shape=(input_leftcontext_lenth, input_maxword_length,), dtype='int32')
#     char_embedding_leftcontext = TimeDistributed(Embedding(input_dim=charvobsize,
#                                                         output_dim=c2v_k,
#                                                         batch_input_shape=(
#                                                         batch_size, input_leftcontext_lenth, input_maxword_length),
#                                                         mask_zero=False,
#                                                         trainable=True,
#                                                         weights=[char_W]))(char_input_leftcontext)
#
#     char_cnn_context = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))
#
#     char_embedding_leftcontext = char_cnn_context(char_embedding_leftcontext)
#     char_embedding_leftcontext = TimeDistributed(GlobalMaxPooling1D())(char_embedding_leftcontext)
#     char_embedding_leftcontext = Dropout(0.25)(char_embedding_leftcontext)
#
#
#     word_input_rightcontext = Input(shape=(input_rightcontext_lenth,), dtype='int32')
#     word_embedding_rightcontext = Embedding(input_dim=wordvocabsize + 1,
#                                output_dim=w2v_k,
#                                input_length=input_rightcontext_lenth,
#                                mask_zero=True,
#                                trainable=True,
#                                weights=[word_W])(word_input_rightcontext)
#     word_embedding_rightcontext = Dropout(0.5)(word_embedding_rightcontext)
#
#     char_input_rightcontext = Input(shape=(input_rightcontext_lenth, input_maxword_length,), dtype='int32')
#     char_embedding_rightcontext = TimeDistributed(Embedding(input_dim=charvobsize,
#                                                         output_dim=c2v_k,
#                                                         batch_input_shape=(
#                                                         batch_size, input_rightcontext_lenth, input_maxword_length),
#                                                         mask_zero=False,
#                                                         trainable=True,
#                                                         weights=[char_W]))(char_input_rightcontext)
#     char_embedding_rightcontext = char_cnn_context(char_embedding_rightcontext)
#     char_embedding_rightcontext = TimeDistributed(GlobalMaxPooling1D())(char_embedding_rightcontext)
#     char_embedding_rightcontext = Dropout(0.25)(char_embedding_rightcontext)
#
#
#     embedding_fragment = concatenate([word_embedding_fragment, char_embedding_fragment], axis=-1)
#     embedding_leftcontext = concatenate([word_embedding_leftcontext, char_embedding_leftcontext], axis=-1)
#     embedding_rightcontext = concatenate([word_embedding_rightcontext, char_embedding_rightcontext], axis=-1)
#
#     LSTM_leftcontext = LSTM(hidden_dim, go_backwards=False, activation='tanh')(embedding_leftcontext)
#
#     LSTM_rightcontext = LSTM(hidden_dim, go_backwards=True, activation='tanh')(embedding_rightcontext)
#
#     BiLSTM_fragment = Bidirectional(LSTM(hidden_dim // 2, activation='tanh'), merge_mode='concat')(embedding_fragment)
#
#     concat = concatenate([LSTM_leftcontext, BiLSTM_fragment, LSTM_rightcontext], axis=-1)
#     concat = Dropout(0.5)(concat)
#     output = Dense(targetvocabsize, activation='softmax')(concat)
#
#     Models = Model([word_input_fragment, word_input_leftcontext, word_input_rightcontext,
#                     char_input_fragment, char_input_leftcontext, char_input_rightcontext], output)
#
#     Models.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.001), metrics=['acc'])
#
#     return Models


def Model2_LSTM_BiLSTM_LSTM(wordvocabsize, targetvocabsize, charvobsize,
                     word_W, char_W,
                     input_fragment_lenth, input_leftcontext_lenth, input_rightcontext_lenth, input_maxword_length,
                     w2v_k, c2v_k,
                     hidden_dim=200, batch_size=32,
                     optimizer='rmsprop'):
    hidden_dim = 100

    word_input_fragment = Input(shape=(input_fragment_lenth,), dtype='int32')
    word_embedding_fragment = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_fragment_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_fragment)
    word_embedding_fragment = Dropout(0.5)(word_embedding_fragment)

    char_input_fragment = Input(shape=(input_fragment_lenth, input_maxword_length,), dtype='int32')
    char_embedding_fragment = TimeDistributed(Embedding(input_dim=charvobsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_fragment_lenth, input_maxword_length),
                               mask_zero=True,
                               trainable=True,
                               weights=[char_W]))(char_input_fragment)

    char_bilstm = Bidirectional(LSTM(30, activation='tanh'), merge_mode='concat')
    char_embedding_fragment = TimeDistributed(char_bilstm)(char_embedding_fragment)
    char_embedding_fragment = Dropout(0.25)(char_embedding_fragment)


    word_input_leftcontext = Input(shape=(input_leftcontext_lenth,), dtype='int32')
    word_embedding_leftcontext = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_leftcontext_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_leftcontext)
    word_embedding_leftcontext = Dropout(0.5)(word_embedding_leftcontext)

    char_input_leftcontext = Input(shape=(input_leftcontext_lenth, input_maxword_length,), dtype='int32')
    char_input_rightcontext = Input(shape=(input_rightcontext_lenth, input_maxword_length,), dtype='int32')

    word_input_rightcontext = Input(shape=(input_rightcontext_lenth,), dtype='int32')
    word_embedding_rightcontext = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_rightcontext_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_rightcontext)
    word_embedding_rightcontext = Dropout(0.5)(word_embedding_rightcontext)


    embedding_fragment = concatenate([word_embedding_fragment, char_embedding_fragment], axis=-1)
    embedding_leftcontext = word_embedding_leftcontext
    embedding_rightcontext = word_embedding_rightcontext

    LSTM_leftcontext = LSTM(hidden_dim, go_backwards=False, activation='tanh')(embedding_leftcontext)

    LSTM_rightcontext = LSTM(hidden_dim, go_backwards=True, activation='tanh')(embedding_rightcontext)

    BiLSTM_fragment = Bidirectional(LSTM(hidden_dim // 2, activation='tanh'), merge_mode='concat')(embedding_fragment)

    concat = concatenate([LSTM_leftcontext, BiLSTM_fragment, LSTM_rightcontext], axis=-1)
    concat = Dropout(0.2)(concat)
    concat_1 = Dense(hidden_dim, activation='tanh')(concat)

    output = Dense(targetvocabsize, activation='softmax')(concat_1)

    Models = Model([word_input_fragment, word_input_leftcontext, word_input_rightcontext,
                    char_input_fragment, char_input_leftcontext, char_input_rightcontext], output)

    Models.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.001), metrics=['acc'])

    return Models

def Model3_LSTM_BiLSTM_LSTM(wordvocabsize, targetvocabsize, charvobsize,
                     word_W, char_W,
                     input_fragment_lenth, input_leftcontext_lenth, input_rightcontext_lenth, input_maxword_length,
                     w2v_k, c2v_k,
                     hidden_dim=200, batch_size=32,
                     optimizer='rmsprop'):
    hidden_dim = 100

    word_input_fragment = Input(shape=(input_fragment_lenth,), dtype='int32')
    word_embedding_fragment = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_fragment_lenth,
                               mask_zero=False,
                               trainable=True,
                               weights=[word_W])(word_input_fragment)
    word_embedding_fragment = Dropout(0.5)(word_embedding_fragment)

    char_input_fragment = Input(shape=(input_fragment_lenth, input_maxword_length,), dtype='int32')
    char_embedding_fragment = TimeDistributed(Embedding(input_dim=charvobsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_fragment_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))(char_input_fragment)

    char_cnn_fragment = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))
    char_embedding_fragment = char_cnn_fragment(char_embedding_fragment)
    char_embedding_fragment = TimeDistributed(GlobalMaxPooling1D())(char_embedding_fragment)
    char_embedding_fragment = Dropout(0.25)(char_embedding_fragment)


    word_input_leftcontext = Input(shape=(input_leftcontext_lenth,), dtype='int32')
    word_embedding_leftcontext = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_leftcontext_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_leftcontext)
    word_embedding_leftcontext = Dropout(0.5)(word_embedding_leftcontext)

    char_input_leftcontext = Input(shape=(input_leftcontext_lenth, input_maxword_length,), dtype='int32')
    char_input_rightcontext = Input(shape=(input_rightcontext_lenth, input_maxword_length,), dtype='int32')

    word_input_rightcontext = Input(shape=(input_rightcontext_lenth,), dtype='int32')
    word_embedding_rightcontext = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_rightcontext_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_rightcontext)
    word_embedding_rightcontext = Dropout(0.5)(word_embedding_rightcontext)

    embedding_fragment = concatenate([word_embedding_fragment, char_embedding_fragment], axis=-1)
    embedding_leftcontext = word_embedding_leftcontext
    embedding_rightcontext = word_embedding_rightcontext

    LSTM_leftcontext = LSTM(hidden_dim, go_backwards=False, activation='tanh')(embedding_leftcontext)
    Rep_LSTM_leftcontext = RepeatVector(input_fragment_lenth)(LSTM_leftcontext)
    LSTM_rightcontext = LSTM(hidden_dim, go_backwards=True, activation='tanh')(embedding_rightcontext)
    Rep_LSTM_rightcontext = RepeatVector(input_fragment_lenth)(LSTM_rightcontext)

    BiLSTM_fragment = Bidirectional(LSTM(hidden_dim // 2, activation='tanh',return_sequences=True), merge_mode='concat')(embedding_fragment)
    context_ADD = add([LSTM_leftcontext, BiLSTM_fragment, LSTM_rightcontext])
    context_subtract_l = subtract([BiLSTM_fragment, LSTM_leftcontext])
    context_subtract_r = subtract([BiLSTM_fragment, LSTM_rightcontext])
    context_average = average([LSTM_leftcontext, BiLSTM_fragment, LSTM_rightcontext])
    context_maximum = maximum([LSTM_leftcontext, BiLSTM_fragment, LSTM_rightcontext])

    embedding_mix = concatenate([embedding_fragment, BiLSTM_fragment,
                                 context_ADD, context_subtract_l, context_subtract_r,
                                 context_average, context_maximum], axis=-1)

    # BiLSTM_fragment = Bidirectional(LSTM(hidden_dim // 2, activation='tanh'), merge_mode='concat')(embedding_fragment)

    decoderlayer1 = Conv1D(50, 1, activation='relu', strides=1, padding='same')(embedding_mix)
    decoderlayer2 = Conv1D(50, 2, activation='relu', strides=1, padding='same')(embedding_mix)
    decoderlayer3 = Conv1D(50, 3, activation='relu', strides=1, padding='same')(embedding_mix)
    decoderlayer4 = Conv1D(50, 4, activation='relu', strides=1, padding='same')(embedding_mix)

    CNNs_fragment = concatenate([decoderlayer1, decoderlayer2, decoderlayer3, decoderlayer4], axis=-1)
    CNNs_fragment = Dropout(0.5)(CNNs_fragment)
    CNNs_fragment = GlobalMaxPooling1D()(CNNs_fragment)

    concat = Dropout(0.3)(CNNs_fragment)


    output = Dense(targetvocabsize, activation='softmax')(concat)

    Models = Model([word_input_fragment, word_input_leftcontext, word_input_rightcontext,
                    char_input_fragment, char_input_leftcontext, char_input_rightcontext], output)

    Models.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.001), metrics=['acc'])

    return Models


def Model_LSTM_BiLSTM_LSTM(wordvocabsize, targetvocabsize, charvobsize,
                     word_W, char_W,
                     input_fragment_lenth, input_leftcontext_lenth, input_rightcontext_lenth, input_maxword_length,
                     w2v_k, c2v_k,
                     hidden_dim=200, batch_size=32,
                     optimizer='rmsprop'):

    word_input_fragment = Input(shape=(input_fragment_lenth,), dtype='int32')
    word_embedding_fragment = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_fragment_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_fragment)
    word_embedding_fragment = Dropout(0.5)(word_embedding_fragment)

    char_input_fragment = Input(shape=(input_fragment_lenth, input_maxword_length,), dtype='int32')
    char_embedding_fragment = TimeDistributed(Embedding(input_dim=charvobsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_fragment_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))(char_input_fragment)

    char_cnn_fragment = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))
    char_embedding_fragment = char_cnn_fragment(char_embedding_fragment)
    char_embedding_fragment = TimeDistributed(GlobalMaxPooling1D())(char_embedding_fragment)
    char_embedding_fragment = Dropout(0.25)(char_embedding_fragment)


    word_input_leftcontext = Input(shape=(input_leftcontext_lenth,), dtype='int32')
    word_embedding_leftcontext = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_leftcontext_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_leftcontext)
    word_embedding_leftcontext = Dropout(0.5)(word_embedding_leftcontext)

    char_input_leftcontext = Input(shape=(input_leftcontext_lenth, input_maxword_length,), dtype='int32')
    char_embedding_leftcontext = TimeDistributed(Embedding(input_dim=charvobsize,
                                                        output_dim=c2v_k,
                                                        batch_input_shape=(
                                                        batch_size, input_leftcontext_lenth, input_maxword_length),
                                                        mask_zero=False,
                                                        trainable=True,
                                                        weights=[char_W]))(char_input_leftcontext)

    char_cnn_context = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_leftcontext = char_cnn_context(char_embedding_leftcontext)
    char_embedding_leftcontext = TimeDistributed(GlobalMaxPooling1D())(char_embedding_leftcontext)
    char_embedding_leftcontext = Dropout(0.25)(char_embedding_leftcontext)


    word_input_rightcontext = Input(shape=(input_rightcontext_lenth,), dtype='int32')
    word_embedding_rightcontext = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_rightcontext_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_rightcontext)
    word_embedding_rightcontext = Dropout(0.5)(word_embedding_rightcontext)

    char_input_rightcontext = Input(shape=(input_rightcontext_lenth, input_maxword_length,), dtype='int32')
    char_embedding_rightcontext = TimeDistributed(Embedding(input_dim=charvobsize,
                                                        output_dim=c2v_k,
                                                        batch_input_shape=(
                                                        batch_size, input_rightcontext_lenth, input_maxword_length),
                                                        mask_zero=False,
                                                        trainable=True,
                                                        weights=[char_W]))(char_input_rightcontext)
    char_embedding_rightcontext = char_cnn_context(char_embedding_rightcontext)
    char_embedding_rightcontext = TimeDistributed(GlobalMaxPooling1D())(char_embedding_rightcontext)
    char_embedding_rightcontext = Dropout(0.25)(char_embedding_rightcontext)


    embedding_fragment = concatenate([word_embedding_fragment, char_embedding_fragment], axis=-1)
    embedding_leftcontext = concatenate([word_embedding_leftcontext, char_embedding_leftcontext], axis=-1)
    embedding_rightcontext = concatenate([word_embedding_rightcontext, char_embedding_rightcontext], axis=-1)

    LSTM_leftcontext = LSTM(hidden_dim, go_backwards=False, activation='tanh')(embedding_leftcontext)

    LSTM_rightcontext = LSTM(hidden_dim, go_backwards=True, activation='tanh')(embedding_rightcontext)

    BiLSTM_fragment = Bidirectional(LSTM(hidden_dim // 2, activation='tanh'), merge_mode='concat')(embedding_fragment)


    concat = concatenate([LSTM_leftcontext, BiLSTM_fragment, LSTM_rightcontext], axis=-1)
    concat = Dropout(0.5)(concat)
    output = Dense(targetvocabsize, activation='softmax')(concat)

    Models = Model([word_input_fragment, word_input_leftcontext, word_input_rightcontext,
                    char_input_fragment, char_input_leftcontext, char_input_rightcontext], output)

    Models.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.001), metrics=['acc'])

    return Models


def Model_LSTM_BiLSTM_LSTM_simul(wordvocabsize, targetvocabsize, charvobsize,
                     word_W, char_W,
                     input_fragment_lenth, input_leftcontext_lenth, input_rightcontext_lenth, input_maxword_length,
                     w2v_k, c2v_k,
                     hidden_dim=200, batch_size=32,
                     optimizer='rmsprop'):

    word_input_fragment = Input(shape=(input_fragment_lenth,), dtype='int32')
    word_embedding_fragment = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_fragment_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_fragment)
    word_embedding_fragment = Dropout(0.5)(word_embedding_fragment)

    char_input_fragment = Input(shape=(input_fragment_lenth, input_maxword_length,), dtype='int32')
    char_embedding_fragment = TimeDistributed(Embedding(input_dim=charvobsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_fragment_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))(char_input_fragment)

    char_cnn_fragment = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))
    char_embedding_fragment = char_cnn_fragment(char_embedding_fragment)
    char_embedding_fragment = TimeDistributed(GlobalMaxPooling1D())(char_embedding_fragment)
    char_embedding_fragment = Dropout(0.25)(char_embedding_fragment)


    word_input_leftcontext = Input(shape=(input_leftcontext_lenth,), dtype='int32')
    word_embedding_leftcontext = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_leftcontext_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_leftcontext)
    word_embedding_leftcontext = Dropout(0.5)(word_embedding_leftcontext)

    char_input_leftcontext = Input(shape=(input_leftcontext_lenth, input_maxword_length,), dtype='int32')
    char_embedding_leftcontext = TimeDistributed(Embedding(input_dim=charvobsize,
                                                        output_dim=c2v_k,
                                                        batch_input_shape=(
                                                        batch_size, input_leftcontext_lenth, input_maxword_length),
                                                        mask_zero=False,
                                                        trainable=True,
                                                        weights=[char_W]))(char_input_leftcontext)

    char_cnn_context = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_leftcontext = char_cnn_context(char_embedding_leftcontext)
    char_embedding_leftcontext = TimeDistributed(GlobalMaxPooling1D())(char_embedding_leftcontext)
    char_embedding_leftcontext = Dropout(0.25)(char_embedding_leftcontext)


    word_input_rightcontext = Input(shape=(input_rightcontext_lenth,), dtype='int32')
    word_embedding_rightcontext = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_rightcontext_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_rightcontext)
    word_embedding_rightcontext = Dropout(0.5)(word_embedding_rightcontext)

    char_input_rightcontext = Input(shape=(input_rightcontext_lenth, input_maxword_length,), dtype='int32')
    char_embedding_rightcontext = TimeDistributed(Embedding(input_dim=charvobsize,
                                                        output_dim=c2v_k,
                                                        batch_input_shape=(
                                                        batch_size, input_rightcontext_lenth, input_maxword_length),
                                                        mask_zero=False,
                                                        trainable=True,
                                                        weights=[char_W]))(char_input_rightcontext)
    char_embedding_rightcontext = char_cnn_context(char_embedding_rightcontext)
    char_embedding_rightcontext = TimeDistributed(GlobalMaxPooling1D())(char_embedding_rightcontext)
    char_embedding_rightcontext = Dropout(0.25)(char_embedding_rightcontext)


    embedding_fragment = concatenate([word_embedding_fragment, char_embedding_fragment], axis=-1)
    embedding_leftcontext = concatenate([word_embedding_leftcontext, char_embedding_leftcontext], axis=-1)
    embedding_rightcontext = concatenate([word_embedding_rightcontext, char_embedding_rightcontext], axis=-1)

    LSTM_leftcontext = Bidirectional(LSTM(hidden_dim, go_backwards=False, activation='tanh'), merge_mode='ave')(embedding_leftcontext)

    LSTM_rightcontext = Bidirectional(LSTM(hidden_dim, go_backwards=True, activation='tanh'), merge_mode='ave')(embedding_rightcontext)

    BiLSTM_fragment = Bidirectional(LSTM(hidden_dim // 2, activation='tanh'), merge_mode='concat')(embedding_fragment)


    concat = concatenate([LSTM_leftcontext, BiLSTM_fragment, LSTM_rightcontext], axis=-1)
    concat = Dropout(0.5)(concat)

    output_2t = Dense(2, activation='softmax', name='2type')(concat)

    output_2t_2input = Dense(100, activation=None)(output_2t)

    concat2 = concatenate([LSTM_leftcontext, BiLSTM_fragment, LSTM_rightcontext, output_2t_2input], axis=-1)
    concat2 = Dropout(0.5)(concat2)

    output = Dense(targetvocabsize, activation='softmax', name='5type')(concat2)

    Models = Model([word_input_fragment, word_input_leftcontext, word_input_rightcontext,
                    char_input_fragment, char_input_leftcontext, char_input_rightcontext], [output_2t, output])

    Models.compile(loss='categorical_crossentropy',
                   loss_weights={'5type': 1., '2type': 0.4},
                   optimizer=optimizers.RMSprop(lr=0.001),
                   metrics=['acc'])

    return Models


def Model_3Level(wordvocabsize, targetvocabsize, charvobsize, posivocabsize,
                     word_W, char_W, posi_W,
                     input_fragment_lenth, input_leftcontext_lenth, input_rightcontext_lenth,
                     input_maxword_length, input_sent_lenth,
                     w2v_k, c2v_k, posi_k,
                     hidden_dim=200, batch_size=32,
                     optimizer='rmsprop'):



    word_input_fragment = Input(shape=(input_fragment_lenth,), dtype='int32')
    word_embedding_fragment = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_fragment_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_fragment)
    word_embedding_fragment = Dropout(0.5)(word_embedding_fragment)

    char_input_fragment = Input(shape=(input_fragment_lenth, input_maxword_length,), dtype='int32')
    char_embedding_fragment = TimeDistributed(Embedding(input_dim=charvobsize,
                               output_dim=c2v_k,
                               batch_input_shape=(batch_size, input_fragment_lenth, input_maxword_length),
                               mask_zero=False,
                               trainable=True,
                               weights=[char_W]))(char_input_fragment)

    char_cnn_fragment = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))
    char_embedding_fragment = char_cnn_fragment(char_embedding_fragment)
    char_embedding_fragment = TimeDistributed(GlobalMaxPooling1D())(char_embedding_fragment)
    char_embedding_fragment = Dropout(0.25)(char_embedding_fragment)


    word_input_leftcontext = Input(shape=(input_leftcontext_lenth,), dtype='int32')
    word_embedding_leftcontext = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_leftcontext_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_leftcontext)
    word_embedding_leftcontext = Dropout(0.5)(word_embedding_leftcontext)

    char_input_leftcontext = Input(shape=(input_leftcontext_lenth, input_maxword_length,), dtype='int32')
    char_embedding_leftcontext = TimeDistributed(Embedding(input_dim=charvobsize,
                                                        output_dim=c2v_k,
                                                        batch_input_shape=(
                                                        batch_size, input_leftcontext_lenth, input_maxword_length),
                                                        mask_zero=False,
                                                        trainable=True,
                                                        weights=[char_W]))(char_input_leftcontext)

    char_cnn_context = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_leftcontext = char_cnn_context(char_embedding_leftcontext)
    char_embedding_leftcontext = TimeDistributed(GlobalMaxPooling1D())(char_embedding_leftcontext)
    char_embedding_leftcontext = Dropout(0.25)(char_embedding_leftcontext)


    word_input_rightcontext = Input(shape=(input_rightcontext_lenth,), dtype='int32')
    word_embedding_rightcontext = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_rightcontext_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_rightcontext)
    word_embedding_rightcontext = Dropout(0.5)(word_embedding_rightcontext)

    char_input_rightcontext = Input(shape=(input_rightcontext_lenth, input_maxword_length,), dtype='int32')
    char_embedding_rightcontext = TimeDistributed(Embedding(input_dim=charvobsize,
                                                        output_dim=c2v_k,
                                                        batch_input_shape=(
                                                        batch_size, input_rightcontext_lenth, input_maxword_length),
                                                        mask_zero=False,
                                                        trainable=True,
                                                        weights=[char_W]))(char_input_rightcontext)
    char_embedding_rightcontext = char_cnn_context(char_embedding_rightcontext)
    char_embedding_rightcontext = TimeDistributed(GlobalMaxPooling1D())(char_embedding_rightcontext)
    char_embedding_rightcontext = Dropout(0.25)(char_embedding_rightcontext)

    word_input_sent = Input(shape=(input_sent_lenth,), dtype='int32')
    word_embedding_sent = Embedding(input_dim=wordvocabsize + 1,
                               output_dim=w2v_k,
                               input_length=input_sent_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[word_W])(word_input_sent)
    word_embedding_sent = Dropout(0.5)(word_embedding_sent)

    word_input_posi = Input(shape=(input_sent_lenth,), dtype='int32')
    word_embedding_posi = Embedding(input_dim=posivocabsize,
                               output_dim=posi_k,
                               input_length=input_sent_lenth,
                               mask_zero=True,
                               trainable=True,
                               weights=[posi_W])(word_input_posi)
    word_embedding_posi = Dropout(0.5)(word_embedding_posi)


    embedding_fragment = concatenate([word_embedding_fragment, char_embedding_fragment], axis=-1)
    embedding_leftcontext = concatenate([word_embedding_leftcontext, char_embedding_leftcontext], axis=-1)
    embedding_rightcontext = concatenate([word_embedding_rightcontext, char_embedding_rightcontext], axis=-1)
    embedding_posi = Dense(50, activation=None)(word_embedding_posi)
    embedding_sent = concatenate([word_embedding_sent, embedding_posi], axis=-1)

    LSTM_leftcontext = LSTM(hidden_dim, go_backwards=False, activation='tanh')(embedding_leftcontext)

    LSTM_rightcontext = LSTM(hidden_dim, go_backwards=True, activation='tanh')(embedding_rightcontext)

    BiLSTM_fragment = Bidirectional(LSTM(hidden_dim // 2, activation='tanh'), merge_mode='concat')(embedding_fragment)

    BiLSTM_sent = Bidirectional(LSTM(hidden_dim // 2, activation='tanh'), merge_mode='concat')(embedding_sent)

    concat = concatenate([LSTM_leftcontext, BiLSTM_fragment, LSTM_rightcontext, BiLSTM_sent], axis=-1)
    concat = Dropout(0.5)(concat)
    output = Dense(targetvocabsize, activation='softmax')(BiLSTM_sent)

    Models = Model([word_input_fragment, word_input_leftcontext, word_input_rightcontext,
                    word_input_posi, word_input_sent,
                    char_input_fragment, char_input_leftcontext, char_input_rightcontext], output)

    Models.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.001), metrics=['acc'])

    return Models