#coding=utf-8

__author__ = 'JIA'

import numpy as np
import pickle
import json
import re
import math


def Seq2frag(file, source_vob, target_vob, target_idex_word, max_context=0, max_fragment = 1):
    '''
        main
        :param file:
        :param source_vob:
        :param target_vob:
        :param target_idex_word:
        :return:
    '''

    sen2list_all, tag2list_all = ReadfromTXT(file, source_vob, target_vob)
    fragment_list, max_context, max_fragment = Lists2Set(sen2list_all, tag2list_all, target_idex_word, max_context, max_fragment)

    return fragment_list, max_context


def ReadSeqs():

    pass


def ReadfromTXT(file, source_vob, target_vob):

    sen2list = []
    tag2list = []
    sen2list_all = []
    tag2list_all = []

    f = open(file, 'r')
    fr = f.readlines()
    for line in fr:

        if line.__len__() <= 1:

            sen2list_all.append(sen2list)
            tag2list_all.append(tag2list)

            continue

        sent = line.strip('\r\n').rstrip('\n').split(' ')

        if not source_vob.__contains__(sent[0]):
            sen2list.append(source_vob["**UNK**"])
        else:
            sen2list.append(source_vob[sent[0]])

        tag2list.append(target_vob[sent[4]])

    f.close()

    return sen2list_all, tag2list_all


def Lists2Set(sen2list_all, tag2list_all, target_idex_word, max_context, max_fragment):
    fragment_list = []

    for id, tag2list in enumerate(tag2list_all):

        target_left = 0
        fragtuples_list = []
        for index, tag in enumerate(tag2list):
            if target_idex_word[tag].__contains__('O'):
                target_left = index
                continue

            else:
                if target_idex_word[tag].__contains__('B-'):
                    target_left = index

                elif target_idex_word[tag].__contains__('I-'):
                    pass

                elif target_idex_word[tag].__contains__('S-'):

                    reltag = target_idex_word[tag][2:]
                    tuple = (0, index+1, index, index+1, index, len(tag2list), reltag)
                    fragtuples_list.append(tuple)

                    flens = max(index+1, len(tag2list)-index)
                    if flens > max_context:
                        max_context = flens

                    target_left = index

                elif target_idex_word[tag].__contains__('E-'):

                    reltag = target_idex_word[tag][2:]
                    tuple = (0, index+1, target_left, index+1, target_left, len(tag2list), reltag)
                    fragtuples_list.append(tuple)

                    flens = max(index+1, len(tag2list)-target_left)
                    if flens > max_context:
                        max_context = flens

                    max_fragment = max(max_fragment, index+1-target_left)

                    target_left = index

                else:
                    print("Seq2frag error !!!!!!!!")

        for tup in fragtuples_list:
            context_left = sen2list_all[id][tup[0]:tup[1]]
            fragment = sen2list_all[id][tup[2]:tup[3]]
            context_right = sen2list_all[id][tup[4]:tup[5]]
            fragment_tag = tup[6]
            fragment_list.append((fragment, fragment_tag, context_left, context_right))

    return fragment_list, max_context, max_fragment


if __name__ == '__main__':
    print([[8]*2] * 3)



