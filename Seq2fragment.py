#coding=utf-8

__author__ = 'JIA'

import numpy as np
import pickle
import json
import re
import random


def Seq2frag(file, source_vob, target_vob, target_idex_word, max_context=0, max_fragment=1, hasNeg=True):


    sen2list_all, tag2list_all = ReadfromTXT(file, source_vob, target_vob)
    print('sen2list_all len = ', len(sen2list_all))
    print('tag2list_all len = ', len(tag2list_all))

    target_count = 5648

    if hasNeg:
        # fragment_list, max_context, max_fragment, target_count = Lists2Set_neg_PartErgodic(sen2list_all, tag2list_all, target_idex_word, max_context, max_fragment)
        fragment_list, max_context, max_fragment, target_count = Lists2Set_neg_Ergodic(sen2list_all, tag2list_all,
                                                                                           target_idex_word,
                                                                                           max_context, max_fragment)

    else:
        fragment_list, max_context, max_fragment = Lists2Set(sen2list_all, tag2list_all, target_idex_word, max_context, max_fragment)
    print('len(fragment_list) = ', len(fragment_list))
    print('target_count -----', target_count)

    return fragment_list, max_context, max_fragment, target_count



def Seq2frag4test(testresult_1Step, testfile, source_vob, target_vob, target_idex_word):

    sen2list_all, tag2list_all = ReadfromTXT(testfile, source_vob, target_vob)
    print('sen2list_all len test = ', len(sen2list_all))
    print('tag2list_all len test = ', len(tag2list_all))

    fragment_list = Lists2Set4test(testresult_1Step, sen2list_all, tag2list_all, target_idex_word)
    # fragment_list = Lists2Set4test_PartErgodic(testresult_1Step, sen2list_all, tag2list_all, target_idex_word)
    # fragment_list = Lists2Set4test_ergodic(sen2list_all, tag2list_all, target_idex_word)

    print('len(fragment_list) test = ', len(fragment_list))

    return fragment_list


def ReadfromTXT(file, source_vob, target_vob):

    ner_count = 0

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
            sen2list = []
            tag2list = []

            continue

        sent = line.strip('\r\n').rstrip('\n').split(' ')

        if not source_vob.__contains__(sent[0]):
            sen2list.append(source_vob["**UNK**"])
        else:
            sen2list.append(source_vob[sent[0]])

        tag2list.append(target_vob[sent[4]])

    f.close()

    return sen2list_all, tag2list_all


def Lists2Set4test(testresult_1Step, sen2list_all, tag2list_all, target_idex_word):
    fragment_list = []

    if len(testresult_1Step) != len(sen2list_all) or len(testresult_1Step) != len(tag2list_all):
        while(1):
            print('error1')

    for pid, ptag2list in enumerate(testresult_1Step):

        if len(ptag2list) != len(sen2list_all[pid]) or len(ptag2list) != len(tag2list_all[pid]):
            while (1):
                print('error2')

        fragtuples_list = []
        index = 0
        while index < len(ptag2list):

            if ptag2list[index] == 'O' or ptag2list[index] == '':
                index += 1
                continue
            elif ptag2list[index] == 'B':
                    target_left = index
                    index += 1
                    while index < len(ptag2list):
                        if ptag2list[index] == 'I':
                            index += 1
                            continue
                        elif ptag2list[index] == 'E':

                            reltag = 'NULL'
                            if 'B-' in target_idex_word[tag2list_all[pid][target_left]] and \
                                'E-' in target_idex_word[tag2list_all[pid][index]]:
                                reltag = target_idex_word[tag2list_all[pid][index]][2:]

                            tuple = (0, index + 1, target_left, index + 1, target_left, len(ptag2list), reltag)
                            fragtuples_list.append(tuple)
                            index += 1
                            break
                        else:

                            break

            elif ptag2list[index] == 'S':

                reltag = 'NULL'
                if 'S-' in target_idex_word[tag2list_all[pid][index]]:
                    reltag = target_idex_word[tag2list_all[pid][index]][2:]

                tuple = (0, index + 1, index, index + 1, index, len(ptag2list), reltag)
                fragtuples_list.append(tuple)
                index += 1
                continue

        for tup in fragtuples_list:
            context_left = sen2list_all[pid][tup[0]:tup[1]]
            fragment = sen2list_all[pid][tup[2]:tup[3]]
            context_right = sen2list_all[pid][tup[4]:tup[5]]
            fragment_tag = tup[6]
            fragment_list.append((fragment, fragment_tag, context_left, context_right))

    return fragment_list


def Lists2Set4test_PartErgodic(testresult_1Step, sen2list_all, tag2list_all, target_idex_word):
    fragment_list = []

    if len(testresult_1Step) != len(sen2list_all) or len(testresult_1Step) != len(tag2list_all):
        while(1):
            print('error1')

    for pid, ptag2list in enumerate(testresult_1Step):

        if len(ptag2list) != len(sen2list_all[pid]) or len(ptag2list) != len(tag2list_all[pid]):
            while (1):
                print('error2')

        fragtuples_list = []
        hasinStart = []
        index = 0
        while index < len(ptag2list):

            if ptag2list[index] == 'O' or ptag2list[index] == '':
                index += 1
                continue
            elif ptag2list[index] == 'B':
                    target_left = index
                    index += 1
                    while index < len(ptag2list):
                        if ptag2list[index] == 'I':
                            index += 1
                            continue
                        elif ptag2list[index] == 'E':

                            reltag = 'NULL'
                            if 'B-' in target_idex_word[tag2list_all[pid][target_left]] and \
                                'E-' in target_idex_word[tag2list_all[pid][index]]:
                                reltag = target_idex_word[tag2list_all[pid][index]][2:]

                            tuple = (0, index + 1, target_left, index + 1, target_left, len(ptag2list), reltag)
                            if tuple not in fragtuples_list:
                                fragtuples_list.append(tuple)
                            index += 1

                            maxlen = 4
                            target_right = index + 1
                            for start in range(max(0, target_left - maxlen), min(len(ptag2list), target_right + maxlen)):

                                if start in hasinStart:
                                    continue
                                else:
                                    hasinStart.append(start)

                                for width in range(1, maxlen + 1):

                                    end = start + width

                                    if end > len(ptag2list):
                                        break

                                    if end - start == 1:
                                        tag = tag2list_all[pid][start]
                                        if target_idex_word[tag].__contains__('S-'):
                                            reltag = target_idex_word[tag][2:]

                                        else:
                                            reltag = 'NULL'

                                    else:
                                        starttag = tag2list_all[pid][start]
                                        endtag = tag2list_all[pid][end - 1]
                                        if target_idex_word[starttag].__contains__('B-') and \
                                                target_idex_word[endtag].__contains__('E-'):
                                            reltag = target_idex_word[starttag][2:]

                                        else:
                                            reltag = 'NULL'

                                    tuple = (0, end, start, end, start, len(ptag2list), reltag)
                                    if tuple not in fragtuples_list:
                                        fragtuples_list.append(tuple)


                            break
                        else:

                            break

            elif ptag2list[index] == 'S':

                reltag = 'NULL'
                if 'S-' in target_idex_word[tag2list_all[pid][index]]:
                    reltag = target_idex_word[tag2list_all[pid][index]][2:]

                tuple = (0, index + 1, index, index + 1, index, len(ptag2list), reltag)
                if tuple not in fragtuples_list:
                    fragtuples_list.append(tuple)
                index += 1

                maxlen = 4
                target_left = index
                target_right = index + 1
                for start in range(max(0, target_left - maxlen), min(len(ptag2list), target_right + maxlen)):

                    if start in hasinStart:
                        continue
                    else:
                        hasinStart.append(start)

                    for width in range(1, maxlen + 1):

                        end = start + width

                        if end > len(ptag2list):
                            break

                        if end - start == 1:
                            tag = tag2list_all[pid][start]
                            if target_idex_word[tag].__contains__('S-'):
                                reltag = target_idex_word[tag][2:]

                            else:
                                reltag = 'NULL'

                        else:
                            starttag = tag2list_all[pid][start]
                            endtag = tag2list_all[pid][end - 1]
                            if target_idex_word[starttag].__contains__('B-') and \
                                    target_idex_word[endtag].__contains__('E-'):
                                reltag = target_idex_word[starttag][2:]

                            else:
                                reltag = 'NULL'

                        tuple = (0, end, start, end, start, len(ptag2list), reltag)
                        if tuple not in fragtuples_list:
                            fragtuples_list.append(tuple)


                continue

        for tup in fragtuples_list:
            context_left = sen2list_all[pid][tup[0]:tup[1]]
            fragment = sen2list_all[pid][tup[2]:tup[3]]
            context_right = sen2list_all[pid][tup[4]:tup[5]]
            fragment_tag = tup[6]
            fragment_list.append((fragment, fragment_tag, context_left, context_right))

    return fragment_list


def Lists2Set4test_ergodic(sen2list_all, tag2list_all, target_idex_word):
    fragment_list = []

    for id, tag2list in enumerate(tag2list_all):

        fragtuples_list = []

        maxlen = 4

        for start in range(0, len(tag2list)):

            for width in range(1, maxlen+1):

                end = start + width

                if end > len(tag2list):
                    break

                if end - start == 1:
                    tag = tag2list[start]
                    if target_idex_word[tag].__contains__('S-'):
                        reltag = target_idex_word[tag][2:]
                    else:
                        reltag = 'NULL'

                else:
                    starttag = tag2list[start]
                    endtag = tag2list[end-1]
                    if target_idex_word[starttag].__contains__('B-') and \
                            target_idex_word[endtag].__contains__('E-'):
                        reltag = target_idex_word[starttag][2:]
                    else:
                        reltag = 'NULL'

                tuple = (0, end, start, end, start, len(tag2list), reltag)
                fragtuples_list.append(tuple)


        for tup in fragtuples_list:
            context_left = sen2list_all[id][tup[0]:tup[1]]
            fragment = sen2list_all[id][tup[2]:tup[3]]
            context_right = sen2list_all[id][tup[4]:tup[5]]
            fragment_tag = tup[6]
            fragment_list.append((fragment, fragment_tag, context_left, context_right))

    return fragment_list


def Lists2Set_neg_PartErgodic(sen2list_all, tag2list_all, target_idex_word, max_context, max_fragment):
    fragment_list = []
    target_count = 0

    for id, tag2list in enumerate(tag2list_all):
        hasinStart = []

        target_left = 0
        fragtuples_list = []
        for index, tag in enumerate(tag2list):

            if target_idex_word[tag] == 'O':
                target_left = index
                continue

            else:
                if target_idex_word[tag].__contains__('B-'):
                    target_left = index

                elif target_idex_word[tag].__contains__('I-'):
                    continue

                else:

                    if target_idex_word[tag].__contains__('S-'):

                        target_left = index
                        target_right = index + 1

                    elif target_idex_word[tag].__contains__('E-'):

                        target_right = index + 1

                    reltag = target_idex_word[tag][2:]
                    tuple_posi = (0, target_right, target_left, target_right, target_left, len(tag2list), reltag)
                    fragtuples_list.append(tuple_posi)

                    flens = max(index + 1, len(tag2list) - target_left)
                    if flens > max_context:
                        max_context = flens
                    max_fragment = max(max_fragment, target_right - target_left)

                    target_count += 1


                    maxlen = 4

                    for start in range(max(0, target_left-maxlen), min(len(tag2list),target_right+maxlen)):

                        if start in hasinStart:
                            continue
                        else:
                            hasinStart.append(start)

                        for width in range(1, maxlen + 1):

                            end = start + width

                            if end > len(tag2list):
                                break

                            if end - start == 1:
                                tag = tag2list[start]
                                if target_idex_word[tag].__contains__('S-'):
                                    reltag = target_idex_word[tag][2:]

                                else:
                                    reltag = 'NULL'

                            else:
                                starttag = tag2list[start]
                                endtag = tag2list[end - 1]
                                if target_idex_word[starttag].__contains__('B-') and \
                                        target_idex_word[endtag].__contains__('E-'):
                                    reltag = target_idex_word[starttag][2:]

                                else:
                                    reltag = 'NULL'

                            tuple = (0, end, start, end, start, len(tag2list), reltag)
                            fragtuples_list.append(tuple)
                            if reltag == 'NULL':
                                fragtuples_list.append(tuple_posi)
                            target_count += 1


        for tup in fragtuples_list:
            context_left = sen2list_all[id][tup[0]:tup[1]]
            fragment = sen2list_all[id][tup[2]:tup[3]]
            context_right = sen2list_all[id][tup[4]:tup[5]]
            fragment_tag = tup[6]
            fragment_list.append((fragment, fragment_tag, context_left, context_right))

    return fragment_list, max_context, max_fragment, target_count


def Lists2Set_neg_Ergodic(sen2list_all, tag2list_all, target_idex_word, max_context, max_fragment):
    fragment_list = []
    target_count = 0

    for id, tag2list in enumerate(tag2list_all):
        hasinStart = []

        target_left = 0
        fragtuples_list = []

        maxlen = 4

        for start in range(0, len(tag2list)):

            for width in range(1, maxlen + 1):

                end = start + width

                if end > len(tag2list):
                    break

                if end - start == 1:
                    tag = tag2list[start]
                    if target_idex_word[tag].__contains__('S-'):
                        reltag = target_idex_word[tag][2:]

                    else:
                        reltag = 'NULL'

                else:
                    starttag = tag2list[start]
                    endtag = tag2list[end - 1]
                    if target_idex_word[starttag].__contains__('B-') and \
                            target_idex_word[endtag].__contains__('E-'):
                        reltag = target_idex_word[starttag][2:]

                    else:
                        reltag = 'NULL'



                tuple = (0, end, start, end, start, len(tag2list), reltag)
                fragtuples_list.append(tuple)
                if reltag != 'NULL':
                    target_count += 1

                max_fragment = max(max_fragment, end - start)
                max_context = len(tag2list)


        for tup in fragtuples_list:
            context_left = sen2list_all[id][tup[0]:tup[1]]
            fragment = sen2list_all[id][tup[2]:tup[3]]
            context_right = sen2list_all[id][tup[4]:tup[5]]
            fragment_tag = tup[6]
            fragment_list.append((fragment, fragment_tag, context_left, context_right))

    return fragment_list, max_context, max_fragment, target_count


def Lists2Set_neg(sen2list_all, tag2list_all, target_idex_word, max_context, max_fragment):
    fragment_list = []


    for id, tag2list in enumerate(tag2list_all):

        target_left = 0
        fragtuples_list = []
        for index, tag in enumerate(tag2list):

            if target_idex_word[tag] == 'O':
                target_left = index
                continue

            else:
                if target_idex_word[tag].__contains__('B-'):
                    target_left = index

                elif target_idex_word[tag].__contains__('I-'):
                    continue

                else:

                    if target_idex_word[tag].__contains__('S-'):

                        target_left = index
                        target_right = index + 1

                    elif target_idex_word[tag].__contains__('E-'):

                        target_right = index + 1

                    reltag = target_idex_word[tag][2:]
                    tuple = (0, target_right, target_left, target_right, target_left, len(tag2list), reltag)
                    fragtuples_list.append(tuple)

                    flens = max(index + 1, len(tag2list) - target_left)
                    if flens > max_context:
                        max_context = flens

                    max_context = len(tag2list)

                    max_fragment = max(max_fragment, target_right - target_left)

                    neg_left = random.choice([-2, -1, 1, 2]) + target_left
                    neg_right = random.choice([-2, -1, 1, 2]) + target_right

                    if neg_left < 0:
                        neg_left = random.choice([0, 1, 2]) + target_left
                    if neg_right > len(tag2list):
                        neg_right = random.choice([0, -1, -2]) + target_left

                    if neg_left >= 0 and neg_right <= len(tag2list):
                        if neg_right > neg_left:
                            tuple = (0, neg_right, neg_left, neg_right, neg_left, len(tag2list), 'NULL')
                            fragtuples_list.append(tuple)
                        else:
                            if neg_right + 3 <= len(tag2list):
                                tuple = (0, neg_right+3, neg_left, neg_right+3, neg_left, len(tag2list), 'NULL')
                                fragtuples_list.append(tuple)
                            elif neg_left - 3 >= 0:
                                tuple = (0, neg_right, neg_left, neg_right, neg_left, len(tag2list), 'NULL')
                                fragtuples_list.append(tuple)



        for tup in fragtuples_list:
            context_left = sen2list_all[id][tup[0]:tup[1]]
            fragment = sen2list_all[id][tup[2]:tup[3]]
            context_right = sen2list_all[id][tup[4]:tup[5]]
            fragment_tag = tup[6]
            fragment_list.append((fragment, fragment_tag, context_left, context_right))

    return fragment_list, max_context, max_fragment


def Lists2Set(sen2list_all, tag2list_all, target_idex_word, max_context, max_fragment):
    fragment_list = []


    for id, tag2list in enumerate(tag2list_all):

        target_left = 0
        fragtuples_list = []
        for index, tag in enumerate(tag2list):

            if target_idex_word[tag] == 'O':
                target_left = index
                continue

            else:
                if target_idex_word[tag].__contains__('B-'):
                    target_left = index

                elif target_idex_word[tag].__contains__('I-'):
                    continue

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
    neg_left = random.randint(-2, 2)
    print(neg_left)
    neg_left = random.randint(-2, 2)
    print(neg_left)
    neg_left = random.randint(-2, 2)
    print(neg_left)
    neg_left = random.randint(-2, 2)
    print(neg_left)
    neg_left = random.randint(-2, 2)
    print(neg_left)
    neg_left = random.randint(-2, 2)
    print(neg_left)



