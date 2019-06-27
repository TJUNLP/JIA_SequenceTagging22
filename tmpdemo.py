



if __name__ == '__main__':

    f = '/Users/shengbinjia/Downloads/KRL_FB15K20K'
    fr1 = f + '/mid2name.txt'
    fr2 = f + '/FB15k_description/entity2id.txt'
    f2 = f + '/fb15k_mid2name.txt'

    fr = open(fr1, 'r')
    fw = open(f2, 'w')
    i = 0
    line = fr.readline()
    middict = {}
    while line:
        i += 1
        print(i)
        ls = line.rstrip('\n').split('\t')
        if len(ls) < 2:
            line = fr.readline()
            continue
        if ls[0] not in middict.keys():
            middict[ls[0]] = [ls[1]]
        else:
            middict[ls[0]] += [ls[1]]
        line = fr.readline()

    fr.close()

    fr = open(fr2, 'r')
    for line in fr.readlines():
        ls = line.rstrip('\n').split('\t')
        mid = ls[0]
        if mid in middict.keys():
            fw.write(mid)
            for v in middict[mid]:
                fw.write('\t' + v)

            fw.write('\n')

        else:
            print(mid)

    fw.close()
    fr.close()


