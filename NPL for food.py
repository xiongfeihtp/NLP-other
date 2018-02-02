# -*- coding: utf-8 -*-

import jieba
import json
import os
path='/Users/xiongfei/Desktop/jason_save'
from gensim.models import word2vec
import string

#cut word and save.txt
list_file = os.listdir(path)
comment_list={}
No_load_list=[]
for i in range(0,len(list_file)):
    file = os.path.join(path,list_file[i])
    if os.path.isfile(file):
        with open(file, 'r') as fp:
            try:
                x = json.load(fp)
                #扩展字典
                comment_list.update(x)
            except Exception as e:
                print("loading error for {}".format(file))
                No_load_list.append(file)

tmp=comment_list.copy()
empty_keyword=[]
for word in tmp.keys():
    if tmp[word]:
        continue
    else:
        empty_keyword.append(word)
        del comment_list[word]

#stopword dic
   #把停用词做成字典
stopwords = {}
fstop = open('chinese_stopword.txt', 'r')
for eachWord in fstop:
    stopwords[eachWord.strip('\n')] = eachWord.strip('\n')
fstop.close()

import re
keys_save=[]
#考虑构造中文的标点集合
#remove_punctuation_map = dict((char, None) for char in string.punctuation)
for item in comment_list.keys():
    keys_save.append(item)
    #将标点考虑在分词的范畴当中，这里是以句子为单位
    for line in comment_list[item]:
        #英文标点
        #no_punctuation =line.strip().translate(remove_punctuation_map)
        #中文标点，只保存特定的符号。
        #no_chinese_punctuation=''.join(re.findall(u'[\u4e00-\u9fff]+', no_punctuation))
        #统一过滤所有标点符号
        #no_punctuation= re.sub("[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", "",line.strip())
        line_result=jieba.cut(line)
        filtered = [w for w in line_result if not w in stopwords]
        #这里是a属性，表示添加写入
        with open('cut_result.txt','a') as fp:
            fp.write(" ".join(filtered))

from jieba import analyse
with open("cut_result.txt") as f:
    tokens=f.read()

tf_idf = analyse.extract_tags
tf_idf(tokens,10)
sentences = word2vec.Text8Corpus("cut_result.txt")  # 加载语料
model = word2vec.Word2Vec(sentences, size=200)  # 默认window=5


