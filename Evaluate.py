import numpy as np
import pickle, codecs


def evaluation_NER_syntaxaware(testresult, resultfile):

    total_predict_right = 0.
    total_predict = 0.
    total_right = 0.
    if resultfile != '':
        fres = codecs.open(resultfile, 'w', encoding='utf-8')
    for sent in testresult:
        ptag = sent[0]
        ttag = sent[1]
        # print('ptag--'+str(ptag))
        # print('ttag--'+str(ttag))

        i = 0
        while i < len(ttag):
            # print('ttag['+str(i)+'] is-'+ttag[i]+'-')

            if ttag[i] == '':
                # print( i, '  --', ttag[i], '--')
                i += 1

            elif ttag[i].__contains__('S-LOC') \
                    or ttag[i].__contains__('S-ORG') \
                    or ttag[i].__contains__('S-PER') \
                    or ttag[i].__contains__('S-MISC'):

                total_right += 1.
                # print(i, ttag[i], 'total_right = ', total_right)
                i += 1

            elif ttag[i].__contains__('B-LOC'):
                j = i+1
                while j < len(ttag):
                    if ttag[j].__contains__('I-LOC'):
                        j += 1
                    elif ttag[j].__contains__('E-LOC'):
                        total_right += 1.
                        # print(i, ttag[i], 'total_right = ', total_right)
                        i = j + 1
                        break
                    else:
                        print('error-LOC', i)
            elif ttag[i].__contains__('B-ORG'):
                j = i + 1
                while j < len(ttag):
                    if ttag[j].__contains__('I-ORG'):
                        j += 1
                    elif ttag[j].__contains__('E-ORG'):
                        total_right += 1.
                        # print(i, ttag[i], 'total_right = ', total_right)
                        i = j + 1
                        break
                    else:
                        print('error-ORG', i)
            elif ttag[i].__contains__('B-PER'):
                j = i + 1
                while j < len(ttag):
                    if ttag[j].__contains__('I-PER'):
                        j += 1
                    elif ttag[j].__contains__('E-PER'):
                        total_right += 1.
                        # print(i, ttag[i], 'total_right = ', total_right)
                        i = j + 1
                        break
                    else:
                        print('error-PER', i)
            elif ttag[i].__contains__('B-MISC'):
                j = i + 1
                while j < len(ttag):
                    if ttag[j].__contains__('I-MISC'):
                        j += 1
                    elif ttag[j].__contains__('E-MISC'):
                        total_right += 1.
                        # print(i, ttag[i], 'total_right = ', total_right)
                        i = j + 1
                        break
                    else:
                        print('error-MISC', i)

            elif ttag[i].__contains__('O') or ttag[i].__contains__('VB') or ttag[i].__contains__('CD') or\
                    ttag[i].__contains__('PREP') or ttag[i].__contains__('PUNC') or \
                    ttag[i].__contains__('DT') or ttag[i].__contains__('JJ'):
                i += 1

            else:
                print('final-error-other', i, '  --'+ttag[i]+'--')
                print(ttag)
        # print('total_right = ', total_right)


        i = 0
        while i < len(ptag):

            if ptag[i] == '':
                # print('ptag', i, '  --'+ttag[i]+'--')
                i += 1

            elif ptag[i].__contains__('S-LOC'):
                total_predict += 1.
                if ttag[i].__contains__('S-LOC'):
                    total_predict_right += 1.
                i += 1

            elif ptag[i].__contains__('B-LOC'):
                j = i+1
                if j == len(ptag):
                  i += 1
                while j < len(ptag):
                    if ptag[j].__contains__('I-LOC'):
                        j +=1
                        if j == len(ptag) - 1:
                            i += 1
                    elif ptag[j].__contains__('E-LOC'):
                        total_predict += 1
                        if ttag[i].__contains__('B-LOC'):
                            k = i+1
                            while k < len(ttag):
                                if ttag[k].__contains__('I-LOC'):
                                    k += 1
                                elif ttag[k].__contains__('E-LOC'):
                                    if j ==k:
                                        total_predict_right +=1
                                    break
                        i = j + 1
                        break
                    else:
                        i = j
                        break

            elif ptag[i].__contains__('S-ORG') :
                total_predict +=1.
                if ttag[i].__contains__('S-ORG'):
                    total_predict_right +=1.
                i += 1

            elif ptag[i].__contains__('B-ORG'):
                j = i+1
                if j == len(ptag):
                  i += 1
                while j < len(ptag):
                    if ptag[j].__contains__('I-ORG'):
                        j +=1
                        if j == len(ptag) - 1:
                            i += 1
                    elif ptag[j].__contains__('E-ORG'):
                        total_predict += 1
                        if ttag[i].__contains__('B-ORG'):
                            k = i+1
                            while k < len(ttag):
                                if ttag[k].__contains__('I-ORG'):
                                    k += 1
                                elif ttag[k].__contains__('E-ORG'):
                                    if j == k:
                                        total_predict_right +=1
                                    break
                        i = j + 1
                        break
                    else:
                        i = j
                        break

            elif ptag[i].__contains__('S-MISC') :
                total_predict +=1.
                if ttag[i].__contains__('S-MISC'):
                    total_predict_right +=1.
                i += 1

            elif ptag[i].__contains__('B-MISC'):
                j = i+1
                if j == len(ptag):
                    i += 1
                while j < len(ptag):

                    if ptag[j].__contains__('I-MISC'):

                        j +=1
                        if j == len(ptag) - 1:
                            i += 1
                    elif ptag[j].__contains__('E-MISC'):

                        total_predict += 1
                        if ttag[i].__contains__('B-MISC'):
                            k = i+1
                            while k < len(ttag):
                                if ttag[k].__contains__('I-MISC'):
                                    k += 1
                                elif ttag[k].__contains__('E-MISC'):
                                    if j == k:
                                        total_predict_right +=1
                                    break
                        i = j + 1
                        break
                    else:
                        i = j
                        break

            elif ptag[i].__contains__('S-PER'):
                total_predict += 1.
                if ttag[i].__contains__('S-PER'):
                    total_predict_right +=1.
                i += 1

            elif ptag[i].__contains__('B-PER'):
                j = i+1
                if j == len(ptag):
                    i += 1
                while j < len(ptag):

                    if ptag[j].__contains__('I-PER'):

                        j = j + 1
                        if j == len(ptag)-1:
                            i += 1

                    elif ptag[j].__contains__('E-PER'):

                        total_predict += 1
                        if ttag[i].__contains__('B-PER'):
                            k = i+1
                            while k < len(ttag):

                                if ttag[k].__contains__('I-PER'):
                                    k += 1
                                elif ttag[k].__contains__('E-PER'):
                                    if j == k:
                                        total_predict_right +=1
                                    break
                        i = j + 1
                        break
                    else:
                        # print('!!!!!!!',j, '---'+ptag[j])
                        # print(ptag)
                        # print(ttag)
                        i = j
                        break

            elif ptag[i].__contains__('O') or ptag[i].__contains__('VB') or ptag[i].__contains__('PREP') or\
                ptag[i].__contains__('PUNC') or ptag[i].__contains__('CD') or\
                     ptag[i].__contains__('DT') or ptag[i].__contains__('JJ'):
                i += 1

            else:
                # print('ptag-error-other', i, '  --'+ptag[i]+'--')
                # print(ptag)
                i += 1
        # print('total_predict_right = ', total_predict_right)
        # print('total_predict = ', total_predict)
        if resultfile != '':
            # if ptag != ttag:
            fres.write(str(total_predict_right) + '\t' + str(total_predict) + '\n' +
                       str(ptag) + '\n' +
                       str(ttag) + '\n')
    if resultfile != '':
        fres.close()

    # print('len(testresult)--= ', len(testresult))
    # print('total_predict_right--= ', total_predict_right)
    # print('total_predict--= ', total_predict)
    # print('total_right--=', total_right)

    P = total_predict_right / float(total_predict) if total_predict != 0 else 0
    R = total_predict_right / float(total_right)
    F = (2 * P * R) / float(P + R) if P != 0 else 0

    return P, R, F, total_predict_right, total_predict, total_right


def evaluation_NER(testresult, resultfile):

    total_predict_right = 0.
    total_predict = 0.
    total_right = 0.
    if resultfile != '':
        fres = codecs.open(resultfile, 'w', encoding='utf-8')
    for sent in testresult:
        ptag = sent[0]
        ttag = sent[1]
        # print('ptag--'+str(ptag))
        # print('ttag--'+str(ttag))

        i = 0
        while i < len(ttag):
            # print('ttag['+str(i)+'] is-'+ttag[i]+'-')

            if ttag[i] == '':
                # print( i, '  --', ttag[i], '--')
                i += 1

            elif ttag[i].__contains__('S-LOC') \
                    or ttag[i].__contains__('S-ORG') \
                    or ttag[i].__contains__('S-PER') \
                    or ttag[i].__contains__('S-MISC'):

                total_right += 1.
                # print(i, ttag[i], 'total_right = ', total_right)
                i += 1

            elif ttag[i].__contains__('B-LOC'):
                j = i+1
                while j < len(ttag):
                    if ttag[j].__contains__('I-LOC'):
                        j += 1
                    elif ttag[j].__contains__('E-LOC'):
                        total_right += 1.
                        # print(i, ttag[i], 'total_right = ', total_right)
                        i = j + 1
                        break
                    else:
                        print('error-LOC', i)
            elif ttag[i].__contains__('B-ORG'):
                j = i + 1
                while j < len(ttag):
                    if ttag[j].__contains__('I-ORG'):
                        j += 1
                    elif ttag[j].__contains__('E-ORG'):
                        total_right += 1.
                        # print(i, ttag[i], 'total_right = ', total_right)
                        i = j + 1
                        break
                    else:
                        print('error-ORG', i)
            elif ttag[i].__contains__('B-PER'):
                j = i + 1
                while j < len(ttag):
                    if ttag[j].__contains__('I-PER'):
                        j += 1
                    elif ttag[j].__contains__('E-PER'):
                        total_right += 1.
                        # print(i, ttag[i], 'total_right = ', total_right)
                        i = j + 1
                        break
                    else:
                        print('error-PER', i)
            elif ttag[i].__contains__('B-MISC'):
                j = i + 1
                while j < len(ttag):
                    if ttag[j].__contains__('I-MISC'):
                        j += 1
                    elif ttag[j].__contains__('E-MISC'):
                        total_right += 1.
                        # print(i, ttag[i], 'total_right = ', total_right)
                        i = j + 1
                        break
                    else:
                        print('error-MISC', i)

            elif ttag[i].__contains__('O'):
                i += 1

            else:
                print('final-error-other', i, '  --'+ttag[i]+'--')
                print(ttag)
        # print('total_right = ', total_right)


        i = 0
        while i < len(ptag):

            if ptag[i] == '':
                # print('ptag', i, '  --'+ttag[i]+'--')
                i += 1

            elif ptag[i].__contains__('S-LOC'):
                total_predict += 1.
                if ttag[i].__contains__('S-LOC'):
                    total_predict_right += 1.
                i += 1

            elif ptag[i].__contains__('B-LOC'):
                j = i+1
                if j == len(ptag):
                  i += 1
                while j < len(ptag):
                    if ptag[j].__contains__('I-LOC'):
                        j +=1
                        if j == len(ptag) - 1:
                            i += 1
                    elif ptag[j].__contains__('E-LOC'):
                        total_predict += 1
                        if ttag[i].__contains__('B-LOC'):
                            k = i+1
                            while k < len(ttag):
                                if ttag[k].__contains__('I-LOC'):
                                    k += 1
                                elif ttag[k].__contains__('E-LOC'):
                                    if j ==k:
                                        total_predict_right +=1
                                    break
                        i = j + 1
                        break
                    else:
                        i = j
                        break

            elif ptag[i].__contains__('S-ORG') :
                total_predict +=1.
                if ttag[i].__contains__('S-ORG'):
                    total_predict_right +=1.
                i += 1

            elif ptag[i].__contains__('B-ORG'):
                j = i+1
                if j == len(ptag):
                  i += 1
                while j < len(ptag):
                    if ptag[j].__contains__('I-ORG'):
                        j +=1
                        if j == len(ptag) - 1:
                            i += 1
                    elif ptag[j].__contains__('E-ORG'):
                        total_predict += 1
                        if ttag[i].__contains__('B-ORG'):
                            k = i+1
                            while k < len(ttag):
                                if ttag[k].__contains__('I-ORG'):
                                    k += 1
                                elif ttag[k].__contains__('E-ORG'):
                                    if j == k:
                                        total_predict_right +=1
                                    break
                        i = j + 1
                        break
                    else:
                        i = j
                        break

            elif ptag[i].__contains__('S-MISC') :
                total_predict +=1.
                if ttag[i].__contains__('S-MISC'):
                    total_predict_right +=1.
                i += 1

            elif ptag[i].__contains__('B-MISC'):
                j = i+1
                if j == len(ptag):
                    i += 1
                while j < len(ptag):

                    if ptag[j].__contains__('I-MISC'):

                        j +=1
                        if j == len(ptag) - 1:
                            i += 1
                    elif ptag[j].__contains__('E-MISC'):

                        total_predict += 1
                        if ttag[i].__contains__('B-MISC'):
                            k = i+1
                            while k < len(ttag):
                                if ttag[k].__contains__('I-MISC'):
                                    k += 1
                                elif ttag[k].__contains__('E-MISC'):
                                    if j == k:
                                        total_predict_right +=1
                                    break
                        i = j + 1
                        break
                    else:
                        i = j
                        break

            elif ptag[i].__contains__('S-PER'):
                total_predict += 1.
                if ttag[i].__contains__('S-PER'):
                    total_predict_right +=1.
                i += 1

            elif ptag[i].__contains__('B-PER'):
                j = i+1
                if j == len(ptag):
                    i += 1
                while j < len(ptag):

                    if ptag[j].__contains__('I-PER'):

                        j = j + 1
                        if j == len(ptag)-1:
                            i += 1

                    elif ptag[j].__contains__('E-PER'):

                        total_predict += 1
                        if ttag[i].__contains__('B-PER'):
                            k = i+1
                            while k < len(ttag):

                                if ttag[k].__contains__('I-PER'):
                                    k += 1
                                elif ttag[k].__contains__('E-PER'):
                                    if j == k:
                                        total_predict_right +=1
                                    break
                        i = j + 1
                        break
                    else:
                        # print('!!!!!!!',j, '---'+ptag[j])
                        # print(ptag)
                        # print(ttag)
                        i = j
                        break

            elif ptag[i].__contains__('O'):
                i += 1

            else:
                # print('ptag-error-other', i, '  --'+ptag[i]+'--')
                # print(ptag)
                i += 1
        # print('total_predict_right = ', total_predict_right)
        # print('total_predict = ', total_predict)
        if resultfile != '':
            # if ptag != ttag:
            fres.write(str(total_predict_right) + '\t' + str(total_predict) + '\n' +
                       str(ptag) + '\n' +
                       str(ttag) + '\n')
    if resultfile != '':
        fres.close()

    # print('len(testresult)--= ', len(testresult))
    # print('total_predict_right--= ', total_predict_right)
    # print('total_predict--= ', total_predict)
    # print('total_right--=', total_right)

    P = total_predict_right / float(total_predict) if total_predict != 0 else 0
    R = total_predict_right / float(total_right)
    F = (2 * P * R) / float(P + R) if P != 0 else 0

    return P, R, F, total_predict_right, total_predict, total_right



def evaluation_NER_Type(testresult, resultfile):

    total_predict_right = 0.
    total_predict = 0.
    total_right = 0.
    if resultfile != '':
        fres = codecs.open(resultfile, 'w', encoding='utf-8')
    for sent in testresult:
        ptag = sent[0]
        ttag = sent[1]
        # print('ptag--'+str(ptag))
        # print('ttag--'+str(ttag))

        i = 0
        while i < len(ttag):
            # print('ttag['+str(i)+'] is-'+ttag[i]+'-')

            if ttag[i] == '':
                # print( i, '  --', ttag[i], '--')
                i += 1

            elif ttag[i].__contains__('LOC'):
                j = i+1
                while j < len(ttag):
                    if ttag[j].__contains__('LOC'):
                        j += 1
                    else:
                        total_right += 1.
                        # print(i, ttag[i], 'total_right = ', total_right)
                        i = j
                        break

            elif ttag[i].__contains__('ORG'):
                j = i + 1
                while j < len(ttag):
                    if ttag[j].__contains__('ORG'):
                        j += 1
                    else:
                        total_right += 1.
                        # print(i, ttag[i], 'total_right = ', total_right)
                        i = j
                        break

            elif ttag[i].__contains__('PER'):
                j = i + 1
                while j < len(ttag):
                    if ttag[j].__contains__('PER'):
                        j += 1
                    else:
                        total_right += 1.
                        # print(i, ttag[i], 'total_right = ', total_right)
                        i = j
                        break

            elif ttag[i].__contains__('MISC'):
                j = i + 1
                while j < len(ttag):
                    if ttag[j].__contains__('MISC'):
                        j += 1
                    else:
                        total_right += 1.
                        # print(i, ttag[i], 'total_right = ', total_right)
                        i = j
                        break

            elif ttag[i].__contains__('O'):
                i += 1

            else:
                print('Type-error-other', i, '  --'+ttag[i]+'--')
                print(ttag)
        # print('total_right = ', total_right)


        i = 0
        while i < len(ptag):

            if ptag[i] == '':
                # print('ptag', i, '  --'+ttag[i]+'--')
                i += 1

            elif ptag[i].__contains__('LOC'):
                j = i
                while j < len(ptag):
                    if ptag[j].__contains__('LOC'):
                        j +=1
                        # if j == len(ptag) - 1:
                        #     i += 1
                    else:
                        total_predict += 1
                        if ttag[i].__contains__('LOC'):
                            if i-1 < 0 or (i-1 >=0 and 'LOC' not in ttag[i-1]):
                                k = i+1
                                while k < len(ttag):
                                    if ttag[k].__contains__('LOC'):
                                        k += 1
                                    else:
                                        if j ==k:
                                            total_predict_right +=1
                                        break
                        i = j
                        break

            elif ptag[i].__contains__('ORG'):
                j = i
                while j < len(ptag):
                    if ptag[j].__contains__('ORG'):
                        j +=1
                        # if j == len(ptag) - 1:
                        #     i += 1
                    else:
                        total_predict += 1
                        if ttag[i].__contains__('ORG'):
                            if i-1 < 0 or (i - 1 >= 0 and 'ORG' not in ttag[i - 1]):
                                k = i+1
                                while k < len(ttag):
                                    if ttag[k].__contains__('ORG'):
                                        k += 1
                                    else:
                                        if j ==k:
                                            total_predict_right +=1
                                        break
                        i = j
                        break

            elif ptag[i].__contains__('MISC'):
                j = i
                while j < len(ptag):
                    if ptag[j].__contains__('MISC'):
                        j += 1
                        # if j == len(ptag) - 1:
                        #     i += 1
                    else:
                        total_predict += 1
                        if ttag[i].__contains__('MISC'):
                            if i-1 < 0 or (i - 1 >= 0 and 'MISC' not in ttag[i - 1]):
                                k = i + 1
                                while k < len(ttag):
                                    if ttag[k].__contains__('MISC'):
                                        k += 1
                                    else:
                                        if j == k:
                                            total_predict_right += 1
                                        break
                        i = j
                        break

            elif ptag[i].__contains__('PER'):
                j = i
                while j < len(ptag):
                    if ptag[j].__contains__('PER'):
                        j += 1
                        # if j == len(ptag) - 1:
                        #     i += 1
                    else:
                        total_predict += 1
                        if ttag[i].__contains__('PER'):
                            if i-1 < 0 or (i - 1 >= 0 and 'PER' not in ttag[i - 1]):
                                k = i + 1
                                while k < len(ttag):
                                    if ttag[k].__contains__('PER'):
                                        k += 1
                                    else:
                                        if j == k:
                                            total_predict_right += 1
                                        break
                        i = j
                        break

            elif ptag[i].__contains__('O'):
                i += 1

            else:
                # print('ptag-error-other', i, '  --'+ptag[i]+'--')
                # print(ptag)
                i += 1
        # print('total_predict_right = ', total_predict_right)
        # print('total_predict = ', total_predict)
        if resultfile != '':
            # if ptag != ttag:
            fres.write(str(total_predict_right) + '\t' + str(total_predict) + '\n' +
                       str(ptag) + '\n' +
                       str(ttag) + '\n')
    if resultfile != '':
        fres.close()

    # print('len(testresult)--= ', len(testresult))
    # print('total_predict_right--= ', total_predict_right)
    # print('total_predict--= ', total_predict)
    # print('total_right--=', total_right)

    P = total_predict_right / float(total_predict) if total_predict != 0 else 0
    R = total_predict_right / float(total_right)
    F = (2 * P * R) / float(P + R) if P != 0 else 0

    return P, R, F, total_predict_right, total_predict, total_right


def evaluation_NER_BIOES(testresult, resultfile):

    total_predict_right = 0.
    total_predict = 0.
    total_right = 0.
    if resultfile != '':
        fres = codecs.open(resultfile, 'w', encoding='utf-8')
    for sent in testresult:
        ptag = sent[0]
        ttag = sent[1]
        # print('ptag--'+str(ptag))
        # print('ttag--'+str(ttag))

        i = 0
        while i < len(ttag):
            # print('ttag['+str(i)+'] is-'+ttag[i]+'-')

            if ttag[i] == '':
                # print( i, '  --', ttag[i], '--')
                i += 1

            elif ttag[i].__contains__('S'):

                total_right += 1.
                # print(i, ttag[i], 'total_right = ', total_right)
                i += 1

            elif ttag[i].__contains__('B'):
                j = i+1
                while j < len(ttag):
                    if ttag[j].__contains__('I'):
                        j += 1
                    elif ttag[j].__contains__('E'):
                        total_right += 1.
                        # print(i, ttag[i], 'total_right = ', total_right)
                        i = j + 1
                        break
                    else:
                        print('error', i)

            elif ttag[i].__contains__('O'):
                i += 1

            else:
                print('BIOESerror-other', i, '  --'+ttag[i]+'--')
                print(ttag)
        # print('total_right = ', total_right)


        i = 0
        while i < len(ptag):

            if ptag[i] == '':
                # print('ptag', i, '  --'+ttag[i]+'--')
                i += 1

            elif ptag[i].__contains__('S'):
                total_predict += 1.
                if ttag[i].__contains__('S'):
                    total_predict_right += 1.
                i += 1

            elif ptag[i].__contains__('B'):
                j = i+1
                if j == len(ptag):
                  i += 1
                while j < len(ptag):
                    if ptag[j].__contains__('I'):
                        j +=1
                        if j == len(ptag):#if j == len(ptag) - 1
                            i += 1
                    elif ptag[j].__contains__('E'):
                        total_predict += 1
                        if ttag[i].__contains__('B'):
                            k = i+1
                            while k < len(ttag):
                                if ttag[k].__contains__('I'):
                                    k += 1
                                elif ttag[k].__contains__('E'):
                                    if j ==k:
                                        total_predict_right +=1
                                    break
                        i = j + 1
                        break
                    else:
                        i = j
                        break

            elif ptag[i].__contains__('O'):
                i += 1

            else:
                # print('ptag-error-other', i, '  --'+ptag[i]+'--')
                # print(ptag)
                i += 1
        # print('total_predict_right = ', total_predict_right)
        # print('total_predict = ', total_predict)
        if resultfile != '':
            # if ptag != ttag:
            fres.write(str(total_predict_right) + '\t' + str(total_predict) + '\n' +
                       str(ptag) + '\n' +
                       str(ttag) + '\n')
    if resultfile != '':
        fres.close()

    # print('len(testresult)--= ', len(testresult))
    # print('total_predict_right--= ', total_predict_right)
    # print('total_predict--= ', total_predict)
    # print('total_right--=', total_right)

    P = total_predict_right / float(total_predict) if total_predict != 0 else 0
    R = total_predict_right / float(total_right)
    F = (2 * P * R) / float(P + R) if P != 0 else 0

    return P, R, F, total_predict_right, total_predict, total_right


def evaluation_NER_BIOES_TYPE(testresult, testresult_type, resultfile):

    total_predict_right = 0.
    total_predict = 0.
    total_right = 0.

    if resultfile != '':
        fres = codecs.open(resultfile, 'w', encoding='utf-8')
    for ti, sent in enumerate(testresult):
        ptag = sent[0]
        ttag = sent[1]
        # print('ptag--'+str(ptag))
        # print('ttag--'+str(ttag))

        i = 0
        while i < len(ttag):
            # print('ttag['+str(i)+'] is-'+ttag[i]+'-')

            if ttag[i] == '':
                # print( i, '  --', ttag[i], '--')
                i += 1

            elif ttag[i].__contains__('S'):

                total_right += 1.
                # print(i, ttag[i], 'total_right = ', total_right)
                i += 1

            elif ttag[i].__contains__('B'):
                j = i+1
                while j < len(ttag):
                    if ttag[j].__contains__('I'):
                        j += 1
                    elif ttag[j].__contains__('E'):
                        total_right += 1.
                        # print(i, ttag[i], 'total_right = ', total_right)
                        i = j + 1
                        break
                    else:
                        print('error', i)

            elif ttag[i].__contains__('O'):
                i += 1

            else:
                print('BIOESerror-other', i, '  --'+ttag[i]+'--')
                print(ttag)
        # print('total_right = ', total_right)


        i = 0
        while i < len(ptag):

            if ptag[i] == '':
                # print('ptag', i, '  --'+ttag[i]+'--')
                i += 1

            elif ptag[i].__contains__('S'):

                if testresult_type[ti][0][i] != 'O':
                    total_predict += 1.

                if ttag[i].__contains__('S'):

                    if testresult_type[ti][0][i] == testresult_type[ti][1][i]:
                        total_predict_right += 1.

                i += 1

            elif ptag[i].__contains__('B'):
                j = i+1
                if j == len(ptag):
                  i += 1
                while j < len(ptag):
                    if ptag[j].__contains__('I'):
                        j += 1
                        if j == len(ptag):#if j == len(ptag) - 1
                            i += 1
                    elif ptag[j].__contains__('E'):
                        tmpcount = 0
                        for k in range(i, j+1):
                            if testresult_type[ti][0][k] != 'O':
                                tmpcount += 1
                        if tmpcount >= ((j+1-i) / 2 - 0.01):
                            total_predict += 1


                        if ttag[i].__contains__('B'):
                            k = i+1
                            while k < len(ttag):
                                if ttag[k].__contains__('I'):
                                    k += 1
                                elif ttag[k].__contains__('E'):
                                    if j ==k:

                                        tmpcount = 0
                                        for k in range(i, j + 1):
                                            if testresult_type[ti][0][k] == testresult_type[ti][1][k]:
                                                tmpcount += 1
                                        if tmpcount > ((j + 1 - i) / 2 - 0.01) :
                                            total_predict_right += 1

                                    break
                        i = j + 1
                        break
                    else:
                        i = j
                        break

            elif ptag[i] == 'O':
                i += 1

            else:
                # print('ptag-error-other', i, '  --'+ptag[i]+'--')
                # print(ptag)
                i += 1
        # print('total_predict_right = ', total_predict_right)
        # print('total_predict = ', total_predict)
        if resultfile != '':
            # if ptag != ttag:
            fres.write(str(total_predict_right) + '\t' + str(total_predict) + '\n' +
                       str(ptag) + '\n' +
                       str(ttag) + '\n')
    if resultfile != '':
        fres.close()

    # print('len(testresult)--= ', len(testresult))
    # print('total_predict_right--= ', total_predict_right)
    # print('total_predict--= ', total_predict)
    # print('total_right--=', total_right)

    P = total_predict_right / float(total_predict) if total_predict != 0 else 0
    R = total_predict_right / float(total_right)
    F = (2 * P * R) / float(P + R) if P != 0 else 0

    return P, R, F, total_predict_right, total_predict, total_right



def evaluation_NER2(testresult):

    total_predict_right = 0.
    total_predict = 0.
    total_right = 0.

    for sent in testresult:
        ptag = sent[0]
        ttag = sent[1]
        # print('ptag--'+str(ptag))
        # print('ttag--'+str(ttag))

        i = 0
        while i < len(ttag):
            # print('ttag['+str(i)+'] is-'+ttag[i]+'-')

            if ttag[i] == '':
                # print( i, '  --', ttag[i], '--')
                i += 1

            elif ttag[i].__contains__('S-LOC') \
                    or ttag[i].__contains__('S-ORG') \
                    or ttag[i].__contains__('S-PER') \
                    or ttag[i].__contains__('S-MISC'):

                total_right += 1.
                # print(i, ttag[i], 'total_right = ', total_right)
                i += 1

            elif ttag[i].__contains__('B-LOC'):
                j = i+1
                while j < len(ttag):
                    if ttag[j].__contains__('I-LOC'):
                        j += 1
                    elif ttag[j].__contains__('E-LOC'):
                        total_right += 1.
                        # print(i, ttag[i], 'total_right = ', total_right)
                        i = j + 1
                        break
                    else:
                        print('error-LOC', i)
            elif ttag[i].__contains__('B-ORG'):
                j = i + 1
                while j < len(ttag):
                    if ttag[j].__contains__('I-ORG'):
                        j += 1
                    elif ttag[j].__contains__('E-ORG'):
                        total_right += 1.
                        # print(i, ttag[i], 'total_right = ', total_right)
                        i = j + 1
                        break
                    else:
                        print('error-ORG', i)
            elif ttag[i].__contains__('B-PER'):
                j = i + 1
                while j < len(ttag):
                    if ttag[j].__contains__('I-PER'):
                        j += 1
                    elif ttag[j].__contains__('E-PER'):
                        total_right += 1.
                        # print(i, ttag[i], 'total_right = ', total_right)
                        i = j + 1
                        break
                    else:
                        print('error-PER', i)
            elif ttag[i].__contains__('B-MISC'):
                j = i + 1
                while j < len(ttag):
                    if ttag[j].__contains__('I-MISC'):
                        j += 1
                    elif ttag[j].__contains__('E-MISC'):
                        total_right += 1.
                        # print(i, ttag[i], 'total_right = ', total_right)
                        i = j + 1
                        break
                    else:
                        print('error-MISC', i)

            elif ttag[i].__contains__('O'):
                i += 1

            else:
                print('error-other', i, '  --'+ttag[i]+'--')
                print(ttag)
        # print('total_right = ', total_right)


        i = 0
        while i < len(ptag):

            if ptag[i] == '':
                # print('ptag', i, '  --'+ttag[i]+'--')
                i += 1

            elif ptag[i].__contains__('S-LOC'):
                total_predict += 1.
                if ttag[i].__contains__('S-LOC'):
                    total_predict_right += 1.
                i += 1

            elif ptag[i].__contains__('B-LOC'):
                j = i+1
                if j == len(ptag):
                  i += 1
                while j < len(ptag):
                    if ptag[j].__contains__('I-LOC'):
                        j +=1
                        if j == len(ptag) - 1:
                            i += 1
                    else:
                        total_predict += 1
                        if ttag[i].__contains__('B-LOC'):
                            k = i+1
                            while k < len(ttag):
                                if ttag[k].__contains__('I-LOC'):
                                    k += 1
                                elif ttag[k].__contains__('E-LOC'):
                                    if j == k:
                                        total_predict_right +=1
                                    break
                        i = j
                        break


            elif ptag[i].__contains__('S-ORG') :
                total_predict +=1.
                if ttag[i].__contains__('S-ORG'):
                    total_predict_right +=1.
                i += 1

            elif ptag[i].__contains__('B-ORG'):
                j = i+1
                if j == len(ptag):
                  i += 1
                while j < len(ptag):
                    if ptag[j].__contains__('I-ORG'):
                        j +=1
                        if j == len(ptag) - 1:
                            i += 1
                    else:
                        total_predict += 1
                        if ttag[i].__contains__('B-ORG'):
                            k = i+1
                            while k < len(ttag):
                                if ttag[k].__contains__('I-ORG'):
                                    k += 1
                                elif ttag[k].__contains__('E-ORG'):
                                    if j == k:
                                        total_predict_right +=1
                                    break
                        i = j
                        break


            elif ptag[i].__contains__('S-MISC') :
                total_predict +=1.
                if ttag[i].__contains__('S-MISC'):
                    total_predict_right +=1.
                i += 1

            elif ptag[i].__contains__('B-MISC'):
                j = i+1
                if j == len(ptag):
                    i += 1
                while j < len(ptag):

                    if ptag[j].__contains__('I-MISC'):

                        j +=1
                        if j == len(ptag) - 1:
                            i += 1
                    else:

                        total_predict += 1
                        if ttag[i].__contains__('B-MISC'):
                            k = i+1
                            while k < len(ttag):
                                if ttag[k].__contains__('I-MISC'):
                                    k += 1
                                elif ttag[k].__contains__('E-MISC'):
                                    if j == k:
                                        total_predict_right +=1
                                    break
                        i = j
                        break


            elif ptag[i].__contains__('S-PER'):
                total_predict += 1.
                if ttag[i].__contains__('S-PER'):
                    total_predict_right +=1.
                i += 1

            elif ptag[i].__contains__('B-PER'):
                j = i+1
                if j == len(ptag):
                    i += 1
                while j < len(ptag):

                    if ptag[j].__contains__('I-PER'):

                        j = j + 1
                        if j == len(ptag)-1:
                            i += 1

                    else:

                        total_predict += 1
                        if ttag[i].__contains__('B-PER'):
                            k = i+1
                            while k < len(ttag):

                                if ttag[k].__contains__('I-PER'):
                                    k += 1
                                elif ttag[k].__contains__('E-PER'):
                                    if j == k:
                                        total_predict_right +=1
                                    break
                        i = j
                        break

            elif ptag[i].__contains__('O'):
                i += 1

            else:
                # print('ptag-error-other', i, '  --'+ptag[i]+'--')
                # print(ptag)
                i += 1
        # print('total_predict_right = ', total_predict_right)
        # print('total_predict = ', total_predict)

    # print('len(testresult)--= ', len(testresult))
    # print('total_predict_right--= ', total_predict_right)
    # print('total_predict--= ', total_predict)
    # print('total_right--=', total_right)

    P = total_predict_right / float(total_predict) if total_predict != 0 else 0
    R = total_predict_right / float(total_right)
    F = (2 * P * R) / float(P + R) if P != 0 else 0

    print('evaluate2---', ' P: ', P, 'R: ', R, 'F: ', F)

    return P, R, F, total_predict_right, total_predict, total_right



if __name__ == "__main__":
    # resultname = "./data/demo/result/biose-loss5-result-15"
    # testresult = pickle.load(open(resultname, 'rb'))
    # P, R, F = evaluavtion_triple(testresult)
    # print(P, R, F)
    sen= ['ab', 'bc', 'cd','de', 'ef']
    for i, str in enumerate(sen):
        print(i,' ', str)
        i += 2
    for i in range(0, sen.__len__()):
        print(i, ' ', sen[i])
        i += 2

