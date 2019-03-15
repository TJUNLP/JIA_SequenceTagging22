# coding:utf-8

from keras.layers.core import Dropout
from keras.layers.merge import concatenate
from keras.layers import TimeDistributed, Input, Bidirectional, Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D
from keras.models import Model
from keras import optimizers
from keras.layers.normalization import BatchNormalization


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

    char_cnn = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid'))

    char_embedding_fragment = char_cnn(char_embedding_fragment)
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
    char_embedding_leftcontext = char_cnn(char_embedding_leftcontext)
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
    char_embedding_rightcontext = char_cnn(char_embedding_rightcontext)
    char_embedding_rightcontext = TimeDistributed(GlobalMaxPooling1D())(char_embedding_rightcontext)
    char_embedding_rightcontext = Dropout(0.25)(char_embedding_rightcontext)


    embedding_fragment = concatenate([word_embedding_fragment, char_embedding_fragment], axis=-1)
    embedding_leftcontext = concatenate([word_embedding_leftcontext, char_embedding_leftcontext], axis=-1)
    embedding_rightcontext = concatenate([word_embedding_rightcontext, char_embedding_rightcontext], axis=-1)

    LSTM_leftcontext = LSTM(hidden_dim, go_backwards=False, activation='tanh')(embedding_leftcontext)

    LSTM_rightcontext = LSTM(hidden_dim, go_backwards=True, activation='tanh')(embedding_rightcontext)

    BiLSTM_fragment = Bidirectional(LSTM(hidden_dim / 2, activation='tanh'), merge_mode='concat')(embedding_fragment)

    concat = concatenate([LSTM_leftcontext, BiLSTM_fragment, LSTM_rightcontext], axis=-1)
    concat = Dropout(0.5)(concat)
    output = Dense(targetvocabsize, activation='softmax')(concat)

    Models = Model([word_input_fragment, word_input_leftcontext, word_input_rightcontext,
                    char_input_fragment, char_input_leftcontext, char_input_rightcontext], output)

    Models.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.001), metrics=['acc'])

    return Models