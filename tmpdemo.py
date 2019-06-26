



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
        ls = line.rstrip('\n').split()
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
        ls = line.rstrip('\n').split()
        mid = ls[0]
        if mid in middict.keys():

            if len(middict[mid]) != 1:
                print(middict[mid])
            fw.write(mid + '\t' + middict[mid] + '\n')

        else:
            print(mid)

    fw.close()
    fr.close()


