from gensim.models import word2vec
import os
import gensim
import jieba
import logging


# 此函数作用是对初始语料进行分词处理后，作为训练模型的语料
def cut_txt(old_files, cut_file):

    for testf in old_files:
        fi = open(testf, 'r')
        sentence = ''
        for line in fi.readlines():
            if line.__len__() <= 1:
                sentence = '\EOS1' + sentence + ' ' + '\EOS2'
                #     .replace('，', '').replace('。', '').replace('？', '').replace('！', '') \
                # .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('（', '').replace('）', '') \
                # .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
                # .replace('’', '')  # 去掉标点符号
                fo = open(cut_file, 'a+', encoding='utf-8')
                fo.write(sentence + '\n')
                fo.close()
                sentence = ''
                continue

            sourc = line.strip('\r\n').rstrip('\n').rstrip('\r').split(' ')
            text = sourc[0]
            new_text = ' '.join(text)
            sentence = sentence + ' \space ' + new_text

        fi.close()

    return cut_file

def model_train(train_file_name, save_model_file):  # model_file_name为训练语料的路径,save_model为保存模型名

    # 模型训练，生成词向量
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(train_file_name)  # 加载语料

    # 第一个参数是训练语料，第二个参数是小于该数的单词会被剔除，默认值为5, 第三个参数是神经网络的隐藏层单元数，默认为100
    model = word2vec.Word2Vec(sentences, min_count=5, size=50, window=4, workers=4, iter=5)

    # model.save(save_model_file)
    model.wv.save_word2vec_format(save_model_name, binary=False)   # 以二进制类型保存模型以便重用

    return model





if __name__ == '__main__':

    save_model_name = './data/w2v/C0NLL2003.NER.c2v.txt'

    trainfile = "./data/CoNLL2003_NER/eng.train.BIOES.txt"
    devfile = "./data/CoNLL2003_NER/eng.testa.BIOES.txt"
    testfile = "./data/CoNLL2003_NER/eng.testb.BIOES.txt"
    files = [trainfile, devfile, testfile]

    cut_file = cut_txt(files, './data/w2v/C0NLL2003.NER.c.tmp.txt')

    # cut_file = cut_txt2('./data/ccks17traindata/', cut_file)
    model_1 = model_train(cut_file, save_model_name)

    # 加载已训练好的模型
    # model_1 = word2vec.Word2Vec.load(save_model_name)
    # 计算两个词的相似度/相关程度
    y1 = model_1.similarity("a", ",")
    print(y1)
    print("-------------------------------\n")

    y3 = model_1.similarity("a", "z")
    print(y3)
    print("-------------------------------\n")


    # 计算某个词的相关词列表
    y2 = model_1.most_similar(",", topn=10)  # 10个最相关的

    for item in y2:
        print(item[0], item[1])
    print("-------------------------------\n")