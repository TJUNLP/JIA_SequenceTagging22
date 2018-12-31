
import tensorflow as k

def testNumberofTAG(files):
    wholeTag = {}
    BIOES = {'B': 0, 'I':0, 'O':0, 'E':0, 'S':0 }
    Type = {'LOC': 0, 'ORG': 0,'PER': 0, 'MISC': 0}
    Pos = {}

    for testf in files:
        f = open(testf, 'r')
        fr = f.readlines()
        for line in fr:
            if line.__len__() <= 1:

                continue

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
                Type[sp[1]] +=1

        f.close()

    for tt in wholeTag.keys():
        print(tt, wholeTag[tt])

    for tt1 in BIOES.keys():
        print(tt1, BIOES[tt1])

    for tt2 in Type.keys():
        print(tt2, Type[tt2])

    for pp in Pos.keys():
        print(pp, Pos[pp])



if __name__ == '__main__':

    trainfile = "./data/CoNLL2003_NER/eng.train.BIOES.txt"
    devfile = "./data/CoNLL2003_NER/eng.testa.BIOES.txt"
    testfile = "./data/CoNLL2003_NER/eng.testb.BIOES.txt"

    testNumberofTAG([trainfile, devfile, testfile])