# -*- coding: utf-8 -*-
# @Time    : 2022/1/15 23:42
# @Author  : zhangweiqi
#! -*- coding: utf-8 -*-


import json
import re
from jieba import analyse
from zhon.hanzi import punctuation as p1
from zhon.pinyin import punctuation as p2
import re
import random
from tqdm import tqdm

re_puncs='['+re.escape(p1+p2+" \r\n")+']'


def cutTextToSent(text:str):
    split_text=re.sub(re_puncs,' ',text).split()#每个句子按标点符号分句
    return split_text

# 此处处理停用词
def getStopWords():
    return set(' ')

def readFromJsonFile(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(f.read())


def writeToJsonFile(path: str, obj, indent=None):
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=indent))

#有些大型json文件，是每一行自成json格式，按行写入的
def readFromJsonFileForLine(path: str,max_line=float('inf')):
    with open(path, "r", encoding="utf-8") as f:
        res=[]
        for i in tqdm(f,ncols=80):
            data=i.strip()
            if len(data)!=0:
                res.append(json.loads(data))
            if len(res)>=max_line:
                return res
        return res


#适用于需要不断追加写入的json文件，f为a+模式打开
#每写一次都打开关闭文件一次，可确保能基本实时保存，但效率低
def appendJsonLine(path: str, obj):
    with open(path, "a+", encoding="utf-8") as f:
        f.write(json.dumps(obj,ensure_ascii=False)+'\n')

def writeToJsonFileForLine(path: str, ls:list):
    with open(path, "a+", encoding="utf-8") as f:
        for i in ls:
            f.write(json.dumps(i,ensure_ascii=False)+'\n')

#根据百分位间隔计算长度分布
def cal_Len_distribution(Lens,begin=0,end=100,step=5):
    Lens=list(Lens)
    Lens.sort()
    total_len=len(Lens)
    for i in range(begin,end+1,step):
        idx=int(total_len*i*0.01)
        idx=min(idx,total_len-1)
        print(f"百分位：{i}%，\tindex：{idx}，\t长度：{Lens[idx]}")


#删除多余空白符。对文本保留自然段（换行），其他空白符替换成空格且只保留一个
def delSpace(st:str):
    ls=st.split('\n')#只有换行特殊处理
    ls=list(filter(lambda x:len(x)>0,ls))
    for i in range(len(ls)):
        ls[i]=" ".join(ls[i].split())#若有多个分割，统一用空格替代
    return "\n".join(ls)
