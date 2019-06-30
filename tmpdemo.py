



if __name__ == '__main__':

    f = '/Users/shengbinjia/Downloads/KRL_FB15K20K'
    fr1 = f + '/FB15K/relation2id.txt'
    fr2 = '/Users/shengbinjia/Downloads/NRE-master/data/RE/relation2id.txt'
    # f2 = f + '/fb15k_mid2name.txt'

    # fr = open(fr1, 'r')
    # # fw = open(f2, 'w')
    # i = 0
    # line = fr.readline()
    # middict = {}
    # while line:
    #     i += 1
    #     print(i)
    #     ls = line.rstrip('\n').split('\t')
    #     # if len(ls) < 2:
        #     line = fr.readline()
        #     continue
    #     if ls[0] not in middict.keys():
    #         middict[ls[0]] = [ls[1]]
    #     else:
    #         middict[ls[0]] += [ls[1]]
    #     line = fr.readline()
    #
    # fr.close()

    fr = open(fr1, 'r')
    line = fr.readline()
    middict = []
    while line:
        ls = line.rstrip('\n').split('\t')
        if ls[0] not in middict:
            middict.append(ls[0])
        line = fr.readline()

    fr.close()
    print(len(middict))

    fr = open(fr2, 'r')
    count = 1
    for line in fr.readlines():
        ls = line.rstrip('\n').split(' ')
        mid = ls[0]
        if mid in middict:
            print(count, mid)
            count += 1

        else:
            print(mid)

    fr.close()


