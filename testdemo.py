
import tensorflow as k

def testNumberofTAG(files):
    wholeTag = {}
    BIOES = {'B': 0, 'I':0, 'O':0, 'E':0, 'S':0 }
    Type = {'LOC': 0, 'ORG': 0,'PER': 0, 'MISC': 0}
    Pos = {}

    PosinEnt = {}

    before1 = {}
    before2 = {}
    after1 = {}
    after2 = {}

    for testf in files:
        f = open(testf, 'r')
        fr = f.readlines()
        start = 0
        for id, line in enumerate(fr):
            if line.__len__() <= 1:
                start = 0
                continue
            start += 1
            sourc = line.strip('\r\n').rstrip('\n').rstrip('\r').split(' ')

            if sourc[1] in Pos.keys():
                Pos[sourc[1]] = Pos[sourc[1]] + 1
            else:
                Pos[sourc[1]] = 1

            if sourc[4] in wholeTag.keys():
                wholeTag[sourc[4]] = wholeTag[sourc[4]] + 1
            else:
                wholeTag[sourc[4]] = 1

            if sourc[4] == 'O':
                BIOES['O'] += 1
            else:
                sp = sourc[4].split('-')
                BIOES[sp[0]] += 1
                Type[sp[1]] += 1

                if sourc[1] in PosinEnt.keys():
                    PosinEnt[sourc[1]] = PosinEnt[sourc[1]] + 1
                else:
                    PosinEnt[sourc[1]] = 1

                if 'S-' in sourc[4] or 'B-' in sourc[4]:
                    if start > 2:
                        chch2 = fr[id-2].strip('\r\n').rstrip('\n').rstrip('\r').split(' ')
                        if chch2[1] in before2.keys():
                            before2[chch2[1]] = before2[chch2[1]] + 1
                        else:
                            before2[chch2[1]] = 1
                    if start > 1:
                        chch1 = fr[id-1].strip('\r\n').rstrip('\n').rstrip('\r').split(' ')
                        if chch1[1] in before1.keys():
                            before1[chch1[1]] = before1[chch1[1]] + 1
                        else:
                            before1[chch1[1]] = 1

                if 'S-' in sourc[4] or 'E-' in sourc[4]:

                    if id + 1 < fr.__len__() and fr[id + 1].__len__() > 1:
                        chch1 = fr[id + 1].strip('\r\n').rstrip('\n').rstrip('\r').split(' ')
                        if chch1[1] in after1.keys():
                            after1[chch1[1]] = after1[chch1[1]] + 1
                        else:
                            after1[chch1[1]] = 1
                    if id + 2 < fr.__len__() and fr[id + 2].__len__() > 1:
                        chch2 = fr[id + 2].strip('\r\n').rstrip('\n').rstrip('\r').split(' ')
                        if chch2[1] in after2.keys():
                            after2[chch2[1]] = after2[chch2[1]] + 1
                        else:
                            after2[chch2[1]] = 1

        f.close()

    for tt in wholeTag.keys():
        print(tt, wholeTag[tt])

    for tt1 in BIOES.keys():
        print(tt1, BIOES[tt1])

    for tt2 in Type.keys():
        print(tt2, Type[tt2])

    for pp in Pos.keys():
        print(pp, Pos[pp])

    print('-----------------PosinEnt')
    PosinEnt = sorted(PosinEnt.items(), key=lambda d: d[1], reverse=True)
    for ent in PosinEnt:
        print(ent)

    print('-----------------before1')
    before1 = sorted(before1.items(), key=lambda d: d[1], reverse=True)
    for be in before1:
        print(be)

    print('-----------------before2')
    before2 = sorted(before2.items(), key=lambda d: d[1], reverse=True)
    for be in before2:
        print(be)

    print('-----------------after1')
    after1 = sorted(after1.items(), key=lambda d: d[1], reverse=True)
    for af in after1:
        print(af)

    print('-----------------after2')
    after2 = sorted(after2.items(), key=lambda d: d[1], reverse=True)
    for af in after2:
        print(af)



if __name__ == '__main__':

    trainfile = "./data/CoNLL2003_NER/eng.train.BIOES.txt"
    devfile = "./data/CoNLL2003_NER/eng.testa.BIOES.txt"
    testfile = "./data/CoNLL2003_NER/eng.testb.BIOES.txt"

    # testNumberofTAG([trainfile, devfile, testfile])

    print([1]* 3)