# -*- coding: utf-8 -*-
# @Time    : 2022/1/15 23:07
# @Author  : zhangweiqi
#! -*- coding: utf-8 -*-
import copy
import os
import random
import string
from itertools import chain
from tqdm import tqdm
import jieba
import torch
from Pinyin2Hanzi import DefaultDagParams
from Pinyin2Hanzi import dag
from Pinyin2Hanzi import simplify_pinyin
from pypinyin import lazy_pinyin
from rouge import Rouge
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import numpy
import math
from nltk.translate import bleu
from collections import OrderedDict
from Config import Config
from NLP_Utils import readFromJsonFile, writeToJsonFile,readFromJsonFileForLine,cal_Len_distribution,writeToJsonFileForLine,cutTextToSent

dagparams = DefaultDagParams()

jieba.lcut('1')
hanzi_punctuation='＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'
NORMAL_TOKENS = {ord(f): ord(t) for f, t in zip(
    u'１２３４５６７８９０ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ',
    u'1234567890abcdefghijklmnopqrstuvwxyz')}

def getPuncs():
    return set(hanzi_punctuation+ string.punctuation+' \n\r')
Puncs=getPuncs()
stopPuncs=set(' ,，”、>》}】:;!?。：；？！.\n\r')

def paddingList(ls:list,val,returnTensor=False):
    ls=ls[:]#不要改变了原list尺寸
    maxLen=max([len(i) for i in ls])
    for i in range(len(ls)):
        ls[i]=ls[i]+[val]*(maxLen-len(ls[i]))
    return torch.tensor(ls,device=Config.device) if returnTensor else ls

def list_join(ele,ls:list):
    Len=len(ls)
    if Len==0:
        return []
    res=list(ls[0])
    for i in ls[1:]:
        if ele is not None:
            res.append(ele)
        res.extend(i)
    return res


#cache对话和文本化后的知识即可，三元组的实体和关系不要分词
def cache_data(tk:BertTokenizer):
    if os.path.exists(Config.train_cache_path):#分词太慢了，分好的可以暂存
        train=readFromJsonFile(Config.train_cache_path)
    else:
        train=readFromJsonFile(Config.train_data_path)
        #训练集cache
        for name,data in train.items():
            for one in tqdm(data,ncols=80,desc=name):#每个对话样本
                kg,ut=one['kg'],one['ut']
                for i in range(len(kg)):#三种知识
                    kg[i]=[baseTokenize(tk,_) for _ in kg[i]]
                assert ut[0][1]==True
                for i in range(len(ut)):#每个对话句子
                    ut[i][0]=baseTokenize(tk,ut[i][0])
                    ut[i][2]=[baseTokenize(tk,_)[:1000] for _ in ut[i][2]]
        writeToJsonFile(Config.train_cache_path,train)
    return train


def cache_kg():#为了方便检索知识图谱的知识（kdconv提供）
    all_kg=readFromJsonFile(Config.kg_path)
    for k,v in all_kg.items():
        norm_k=k.replace('(','（').split('（')[0]
        norm_k='_'.join(norm_k.split())#空格等统一替换成下划线字符
        norm_k=norm_k.translate(NORMAL_TOKENS).lower()
        rel=list(OrderedDict([(_[1],None) for _ in v]).keys())
        v=[set(norm_k),rel,v]
        all_kg[k]=v
    print(f'kg_len：{len(all_kg)}')
    Config.all_kg=all_kg

#输入模型生成的实体，用字的交集求数据库上对应实体
#返回（原始的实体名，[所有关系（去重后）], [三元组列表]）
def match_entity(inp:str):
    #inp默认是模型生成的，以下划线作为分隔符的
    #先看看能不能完整匹配
    inp=inp.strip('_')
    temp=inp.replace('_',' ')
    if temp in Config.all_kg:
        return temp,Config.all_kg[temp][1],Config.all_kg[temp][2]
    #无法精确匹配，就归一化然后模糊匹配
    norm_inp=inp.replace('(','（').split('（')[0]
    norm_inp='_'.join(norm_inp.split())#空格等统一替换成下划线字符
    norm_inp=norm_inp.translate(NORMAL_TOKENS).lower()
    norm_inp_set=set(norm_inp)
    ans=[]
    for k,(k_set,rel,kg) in Config.all_kg.items():
        num=len(norm_inp_set&k_set)
        p,r=num/max(len(norm_inp_set),1),num/max(len(k_set),1)#防止分母为0
        ans.append([(p+r),(k,rel,kg)])
    ans.sort(key=lambda x:x[0],reverse=True)
    return ans[0][1]

def match_entity_ls(entity_ls:list,disable_tqdm=True):
    entity_bak=entity_ls
    #优化，可以去重，只匹配独一无二的
    entity_set=set(entity_ls)
    if '无' in entity_set:
        entity_set.remove('无')
    entity_dict=dict()
    for _ in tqdm(entity_set,ncols=80,desc='实体链接……',disable=disable_tqdm):
        entity_dict[_]=match_entity(_)[0]
    entity_ls=['无' if _=='无' else entity_dict[_] for _ in entity_bak]
    return entity_ls


"""
输入：[conv] history [kg] 文本化后的三元组知识 [SEP]
输出：[gen_response] response [SEP]
"""
def encodeText(history:list,profiles:list,tk:BertTokenizer,history_max_num=Config.max_ut_num,returnTensor=False,shuffle_profile=False,profile_maxlen=200,end_p1=True):
    history=history[-history_max_num:]#限制句子条数
    if type(history[0]) is str:#需要分词
        history=[baseTokenize(tk,_) for _ in history]
        profiles=[baseTokenize(tk,_) for _ in profiles]
    if shuffle_profile:
        random.shuffle(profiles)
    profiles=truncate_lists_drop_tail(profiles,profile_maxlen)
    profile_len=sum(len(_) for _ in profiles)
    history=truncate_lists(history,Config.enc_maxlen-profile_len)#history根据profiles最终长度动态调整
    #根据是生成还是打分，分配末尾开始角色为p1或p2，分配role token，所以先反过来
    end_role,other_role=(Config.p1_token,Config.p2_token) if end_p1 else (Config.p2_token,Config.p1_token)
    history=[([other_role] if i&1 else [end_role])+history[::-1][i] for i in range(len(history))]
    history.reverse()
    profiles,history=list_join('/',profiles),list_join(None,history)
    input_ids=['[conv]']+history+['[kg]']+profiles+[tk.sep_token]
    if returnTensor:
        input_ids=torch.tensor(tk.convert_tokens_to_ids(input_ids),device=Config.device)
    return input_ids

"""
输入历史和要预测的实体，转为模型输入格式
输入：[conv] history [SEP]
输出：[gen_entity] 实体 [SEP]
"""
def encodeEntityPred(history:list,entity:str,tk:BertTokenizer,history_max_num=Config.max_ut_num,returnTensor=False,profile_maxlen=200,end_p1=True):
    history=history[-history_max_num:]#限制句子条数
    if type(history[0]) is str:#需要分词
        history=[baseTokenize(tk,_) for _ in history]
    if entity is None:
        entity='无'
    entity=entity.replace('(','（').split('（')[0]#除去实体括号影响
    entity=baseTokenize(tk,entity)[:Config.max_dec_len]#实体都当场分词
    history=encodeText(history,[],tk,history_max_num=history_max_num,profile_maxlen=profile_maxlen,end_p1=end_p1)
    history.pop(-2)
    entity=['[gen_entity]']+entity+[tk.sep_token]
    if returnTensor:
        history=torch.tensor(tk.convert_tokens_to_ids(history),device=Config.device)
        entity=torch.tensor(tk.convert_tokens_to_ids(entity),device=Config.device)
    return history,entity


"""
给定history、实体和候选关系（只给单个实体），预测response所用关系
输入：[conv] history [entity] 实体名 [relation] 实体的所有关系 [SEP]
输出：[gen_relation] 预测要用的所有关系 [SEP]
"""
def encodeRelPred(history:list,kg:list,e_r:dict,tk:BertTokenizer,history_max_num=Config.max_ut_num,returnTensor=False,shuffle_profile=False,profile_maxlen=200,end_p1=True):
    history=history[-history_max_num:]#限制句子条数
    if type(history[0]) is str:#需要分词
        history=[baseTokenize(tk,_) for _ in history]
    entity,rel=kg#预测关系时无需除去实体括号影响
    #需要预测的关系间的顺序，与在关系候选中的顺序要一致
    e_r=e_r[entity]
    e_r=[[_,_ in rel] for _ in e_r]
    if shuffle_profile:
        random.shuffle(e_r)
    input_rel='/'.join(_[0] for _ in e_r)
    output_rel='/'.join(_[0] for _ in filter(lambda x:x[1],e_r))
    entity,input_rel,output_rel=baseTokenize(tk,entity),baseTokenize(tk,input_rel),baseTokenize(tk,output_rel)
    profile=entity+['[relation]']+input_rel
    output_rel=['[gen_relation]']+output_rel[:Config.max_dec_len]+['[SEP]']
    input_ids=encodeText(history,[profile],tk,history_max_num=history_max_num,profile_maxlen=profile_maxlen,end_p1=end_p1)
    for idx,token in enumerate(input_ids):
        if token=='[kg]':
            input_ids[idx]='[entity]'
            break
    return input_ids,output_rel


def union_triplet(triplet):
    if len(triplet)==0:
        return []
    #先处理Information
    temp=[]
    ans=[]
    for i in triplet:
        temp.append(tuple(str(_) for _ in i))
    triplet=temp
    #按前两个合并
    dt=OrderedDict()
    for i in triplet:
        if i[:2] not in dt:
            dt[i[:2]]=[]
        dt[i[:2]].append(i[2])
    temp=[[k[0],f'{k[1]}是{"、".join(v)}' if k[1] not in {'评论','Information'} else f'{k[1]}是“{"、".join(v)}”'] for k,v in dt.items()]
    #再按第一个分组
    dt=OrderedDict()
    for name,attr in temp:
        if name not in dt:
            dt[name]=[]
        dt[name].append(attr)
    for k,v in dt.items():
        v.sort(key=lambda x:len(x))
        ans.append(f'{k}的{"，".join(v)}')
    return list(set(ans))

#拼音转汉字
def pinyin2hanzi(pinyin_ls:list,return_num=5):
    pinyin_ls=[simplify_pinyin(_) for _ in pinyin_ls]
    result = dag(dagparams,pinyin_ls, path_num=return_num)
    return [(item.score, item.path) for item in result]
def is_hanzi(word):
    for ch in word:
        if not ('\u4e00'<=ch<= '\u9fff'):
            return False
    return True

fuzzy_dict={'sm': {'s': 'sh', 'sh': 's', 'c': 'ch', 'ch': 'c', 'z': 'zh', 'zh': 'z', 'l': 'r', 'n': 'l', 'f': 'h', 'h': 'f', 'r': 'l'}, 'ym': {'an': 'ang', 'ang': 'an', 'en': 'eng', 'eng': 'en', 'in': 'ing', 'ing': 'in', 'ian': 'iang', 'iang': 'ian', 'uan': 'uang', 'uang': 'uan'}, 'all': {'fa': 'hua', 'hua': 'fa', 'fan': 'huan', 'huan': 'fan', 'fang': 'huang', 'huang': 'fang', 'fei': 'hui', 'hui': 'fei', 'fen': 'hun', 'hun': 'fen', 'feng': 'hong', 'hong': 'feng', 'fo': 'huo', 'huo': 'fo', 'fu': 'hu', 'hu': 'fu'}}
#汉字转拼音，以fuzzy_prob的概率进行模糊噪声处理
#返回处理后的拼音，已经处理后是否和一开始相同
def hanzi2pinyin(inp:str,fuzzy_prob=0.0):
    def start_with_fuzzy(py:str):#是否以某个模糊声母开头，返回该声母
        for sm in fuzzy_dict['sm']:
            if py.startswith(sm) and (not py.startswith(fuzzy_dict['sm'][sm])):
                return sm
        return False
    def end_with_funzzy(py:str):
        for ym in fuzzy_dict['ym']:
            if py.endswith(ym) and (not py.endswith(fuzzy_dict['ym'][ym])):
                return ym
        return False
    all=lazy_pinyin(inp,errors='ignore')
    #print(all)
    ans=[]
    for py in all:
        if py=='n':
            py='en'
        replace=py
        fuzzy_sm,fuzzy_ym=start_with_fuzzy(py),end_with_funzzy(py)
        #print(fuzzy_sm,fuzzy_ym)
        if py in fuzzy_dict:#整个包含
            replace=fuzzy_dict[py]
        elif fuzzy_ym:#优先替换韵母，因为韵母通用些
            replace=py[:-len(fuzzy_ym)]+fuzzy_dict['ym'][fuzzy_ym]
        elif fuzzy_sm:
            replace=fuzzy_dict['sm'][fuzzy_sm]+py[len(fuzzy_sm):]
        #print(py,replace)
        if random.random()<=fuzzy_prob and len(pinyin2hanzi([replace]))>0:#小概率才替换，也要先判断这个拼音是否合法
            py=replace
        ans.append(py)
    #print(ans)
    return ans,tuple(ans)!=tuple(all)


#对一个具体的词进行增强操作
#fuzzy_prob为每个位置进行拼音替换，或者【插入、删除、复制】的概率
#TODO 加入交换
def fuzzy_word(word:str,fuzzy_prob=0.0):
    #若全为中文字符，先统一先过一遍拼音
    if is_hanzi(word):
        py,has_changed=hanzi2pinyin(word,fuzzy_prob=fuzzy_prob)
        #print(py,has_changed)
        if has_changed or random.random()<=fuzzy_prob:#前后有变化，或者大于一定概率才转中文，因为即使没变化，转中文也有误差
            word=pinyin2hanzi(py,return_num=3)
            word=''.join(random.choice(word)[1])
    #print(word)
    char_ls=['']+list(word)#添加空字符方便开头插入
    ans=[]
    for char in char_ls:
        if random.random()<=fuzzy_prob:
            #每个位置进行插入、删除、复制其中一种
            operate=numpy.random.choice(['insert','del','copy'],p=[0.5,0.25,0.25])
            if operate=='insert':#随机插入标点符号
                char=char+numpy.random.choice(['。','，','？','.',',','?'],p=[0.35,0.3,0.2,0.05,0.05,0.05])
            elif operate=='del':
                char=''
            elif operate=='copy':
                char=char+char
            else:
                exit('出错')
            #print(operate)
        ans.append(char)
    #print(ans)
    return ''.join(ans)

#对一个句子进行增强
def fuzzy_sent(sent,fuzzy_prob=0):
    if type(sent) is list:#已经分词了的，重新用jieba分词
        sent=''.join(sent)
    if fuzzy_prob==0:
        return sent
    sent=jieba.lcut(sent)
    ans=[fuzzy_word(_,fuzzy_prob) for _ in sent]
    return ''.join(ans)


first_sent_pattern=[
'你知道{}吗',
'请问你知道{}吗',
'{}你知道吧',
'{}你应该知道吧',
'{}你知不知道',
'你好，知道{}吗',
'你好，你知道{}吗',
'对于{}你知道得多不多',
'你了解{}吗',
'了解{}吗',
'你对{}有了解吗',
'你对{}了解的多不多',
'对于{}你有多少了解',
'你听过{}吗',
'你听说过{}吗',
'有没有听过{}',
'你好，听说过{}吗',
'你看过{}吗',
'嗨，咱们能聊聊{}吗',
'你好，有兴趣聊一聊{}吗',
'你认识{}吗'
]
def simul_first_sent(num=100):
    entity_list=list(Config.all_kg.keys())
    Len=len(first_sent_pattern)
    ans=[]#（首句，所用实体）
    for i in range(num):
        pt=first_sent_pattern[i%Len]
        entity=random.choice(entity_list)
        sent=pt.format(entity)
        sent=sent+numpy.random.choice(['','？','?'],p=[0.4,0.4,0.2])
        ans.append([sent,entity])
    return ans

#当对话的第二句为空实体时，给定第一句和e_r，返回要生成的实体关系
#[(实体，关系)]
def get_entity_rel_for_first_sent(first_sent,e_r:dict):
    def matching(entity:str,sent:str,threshold=0.75):
        entity=entity.replace('(','（').split('（')[0]
        entity=entity.translate(NORMAL_TOKENS).lower()
        sent=sent.translate(NORMAL_TOKENS).lower()
        entity,sent=''.join(cutTextToSent(entity)),''.join(cutTextToSent(sent))#除去标点符号影响
        if len(entity)==0 or len(sent)==0:
            return False
        if (entity in sent) or (sent in entity):#包含就一定匹配
            return True
        if len(entity)<=3 or len(sent)<=3:#太短的，又无法直接重合，就认定不相似
            return False
        rouge = Rouge(exclusive=False)
        sent=[" ".join(i) for i in [sent]]#基于字
        entity=[" ".join(i) for i in [entity]]
        rg=rouge.get_scores(sent,entity,avg=entity,ignore_empty=entity)
        #print(rg)
        return rg['rouge-l']['r']>=threshold
    if type(first_sent) is list:
        first_sent=''.join(first_sent)
    assert type(first_sent) is str
    ans=[]
    for entity,rel in e_r.items():
        if matching(entity,first_sent,threshold=0.75):
            #选择一个关系
            if 'Information' in rel:
                ans.append((entity,['Information']))
            else:
                ans.append((entity,[random.choice(rel)]))
    return ans


class seq2seqData(Dataset):
    #传入句子对列表
    def __init__(self,textLs:dict,tk:BertTokenizer):
        super().__init__()
        self.tk=tk
        self.origin_data=textLs
        self.aug_data=None#实际使用的，数据增强后的data
        print('参与训练的数据集：',self.origin_data.keys())

    def __len__(self):
        return Config.data_size

    #每一轮重新对数据进行数据增强（因为数据增强会影响长度，分块shuffle需要知道所有数据的长度）
    def data_aug(self):
        self.aug_data=[]
        for name,data in self.origin_data.items():
            #data=data[:100]
            for one in tqdm(data,ncols=80,desc=name):#每个对话样本
                kg=one['kg']
                ut=[_[0] for _ in one['ut']]
                for idx,(u,drop,u_kg,u_entity_rel) in enumerate(one['ut']):
                    if not drop:#需要优化
                        profiles=list(u_kg)#知识在每句话里面，以此作为知识
                        if random.random()<=0.8:
                            history=ut[:idx]
                        else:#history小概率才选取有数据增强的
                            if name in {'hw'}:#验证集自带有错的数据
                                fuzzy_ut=kg[2]
                            else:
                                fuzzy_ut=[fuzzy_sent(_,Config.fuzzy_prob) for _ in ut]
                                fuzzy_ut=[baseTokenize(self.tk,_) for _ in fuzzy_ut]
                            history=fuzzy_ut[:idx]
                        response=ut[idx]
                        #对话生成部分
                        input_ids=encodeText(history,profiles,self.tk,shuffle_profile=True,profile_maxlen=Config.profile_maxlen[name])
                        decoder_input_ids=['[gen_response]']+[Config.p2_token]+response[:Config.max_dec_len]+[self.tk.sep_token]
                        self.aug_data.append([input_ids,decoder_input_ids])
                        #实体预测和关系预测部分
                        e_r=one['e_r']
                        if len(u_entity_rel)==0:#需要预测空实体
                            input_ids,decoder_input_ids=encodeEntityPred(history,None,self.tk)
                            self.aug_data.append([input_ids,decoder_input_ids])
                        for entity,rel in u_entity_rel:
                            #实体
                            input_ids,decoder_input_ids=encodeEntityPred(history,entity,self.tk)
                            self.aug_data.append([input_ids,decoder_input_ids])
                            #关系
                            input_ids,decoder_input_ids=encodeRelPred(history,[entity,rel],e_r,self.tk,shuffle_profile=True)
                            self.aug_data.append([input_ids,decoder_input_ids])

        print('input_ids长度分布：')
        cal_Len_distribution([len(_[0]) for _ in self.aug_data])
        print('deocder_input_ids长度分布：')
        cal_Len_distribution([len(_[1]) for _ in self.aug_data])
        print('\n滑窗后数据量：',len(self.aug_data))
        # self.aug_data=[[''.join(_[0]),''.join(_[1])] for _ in self.aug_data]
        # writeToJsonFile('data/encode好.json',self.aug_data,indent=4)


    #耗时操作在此进行，可用上多进程
    def __getitem__(self, item):
        now=self.aug_data[item]
        return [self.tk.convert_tokens_to_ids(i) for i in now]

    @classmethod
    def collate(cls,batch):
        res=[]
        for i in range(2):
            now=[_[i] for _ in batch]
            res.append(now)
        res=[paddingList(_,0,True) for _ in res]
        return res

unionList=lambda ls:list(chain(*ls))#按元素拼接
splitList=lambda x,bs:[x[i:i+bs] for i in range(0,len(x),bs)]#按bs切分

#%%
#想要修改DataLoader逻辑，每个epoch分块shuffle
#要么修改类的__iter__函数（魔法函数，修改对象的不管用），要么暴力引用进行shuffle
#sortBsNum：原序列按多少个bs块为单位排序，可用来增强随机性
#比如如果每次打乱后都全体一起排序，那每次都是一样的
def blockShuffle(data:list,bs:int,sortBsNumForInput):
    random.shuffle(data)#先打乱
    if (not Config.blockShuffle_gen_data) or Config.loss_weights['gen']==0:#不需要分块shuffle，或者生成的loss权值为0
        return data
    tail=len(data)%bs#计算碎片长度
    tail=[] if tail==0 else data[-tail:]
    data=data[:len(data)-len(tail)]
    assert len(data)%bs==0#剩下的一定能被bs整除
    sortBsNumForInput=len(data)//bs if (sortBsNumForInput is None or sortBsNumForInput==0) else sortBsNumForInput#为None就是整体排序

    data=splitList(data,sortBsNumForInput*bs)
    data=[sorted(i,key=lambda x:len(x[0]),reverse=True) for i in data]#先按第一种大块先用输出长度进行降排序
    data=unionList(data)

    data=splitList(data,bs)#最后，按bs分块
    random.shuffle(data)#块间打乱
    data=unionList(data)+tail
    print('分块shuffle！')
    #writeToJsonFileForLine('data/分块后输入.json',data)
    return data

#每轮迭代重新分块shuffle数据的DataLoader
class blockShuffleDataLoader(DataLoader):
    def __init__(self, dataset: Dataset,sortBsNumForInput,**kwargs):
        super().__init__(dataset,**kwargs)#父类的参数传过去
        self.sortBsNumForInput=sortBsNumForInput


    def __iter__(self):
        #先数据增强
        self.dataset.data_aug()
        #分块shuffle
        self.dataset.aug_data=blockShuffle(self.dataset.aug_data,self.batch_size,self.sortBsNumForInput)
        if self.persistent_workers and self.num_workers > 0:
            if self._iterator is None:
                self._iterator = self._get_iterator()
            else:
                self._iterator._reset(self)
            return self._iterator
        else:
            return self._get_iterator()


#bert输入O(1)时间复杂度的截断
def truncate_pair(a:list,b:list,maxLen:int):
    assert maxLen>=0
    len2=maxLen//2#若为奇数，更长部分给左边
    len1=maxLen-len2
    #一共就a超长与否，b超长与否，组合的四种情况
    if len(a)+len(b)>maxLen:#需要截断
        if len(a)<=len1 and len(b)>len2:
            b=b[:maxLen-len(a)]
        elif len(a)>len1 and len(b)<=len2:
            a=a[:maxLen-len(b)]
        elif len(a)>len1 and len(b)>len2:
            a=a[:len1]
            b=b[:len2]
    return a,b

def truncate_lists_drop_tail(ls:list,maxLen:int):
    ls=copy.deepcopy(ls)#不要破坏原始的
    ls.sort(key=lambda x:len(x))
    assert maxLen>=0
    sumLen=sum([len(i) for i in ls])
    if sumLen<=maxLen:
        return ls
    ans=[]
    now_len=0
    for i in ls:
        if now_len+len(i)<maxLen:
            now_len+=len(i)
            ans.append(i)
        else:
            ans.append(i[:maxLen-now_len])
            break
    return ans

def truncate_lists(ls:list,maxLen:int):
    ls=copy.deepcopy(ls)#不要破坏原始的
    assert maxLen>=0
    sumLen=sum([len(i) for i in ls])
    while sumLen>maxLen:
        id=-1#记录最长的那个id
        nowMaxLen=-1
        for i in range(len(ls)):
            if len(ls[i])>=nowMaxLen:#>=优先截短后面的
                nowMaxLen=len(ls[i])
                id=i
        ls[id].pop()
        sumLen-=1
    return ls

#为某些值加入滑动平均
class EMA():
    def __init__(self,decay):
        self.decay=decay
        self.val=0
        self.beta_exp=1
        self.ema_val=None#滑动平均后的值

    def update(self, now_val):
        self.val=self.val*self.decay+(1-self.decay)*now_val
        self.beta_exp*=self.decay
        self.ema_val=self.val/(1-self.beta_exp)
        return self.ema_val

#适应于多任务下，含梯度累积、指数滑动平均的loss多显示
class Multi_accum_ema_loss():
    def __init__(self,weights:list):
        self.weight_list=weights#每个loss的权重
        self.accum_loss_list=[0]*len(weights)#用于梯度累积过程累加的loss
        self.EMA_list=[EMA(Config.loss_momentum) for i in range(len(weights))]#滑动平均后用于显示的loss
        self.total_weighted_loss=None#分别滑动平均，再加权平均后的loss

    def add_accum_loss(self,loss_list:list):
        assert len(self.accum_loss_list)==len(loss_list)
        for i in range(len(self.accum_loss_list)):
            self.accum_loss_list[i]+=loss_list[i]

    #更新并获取滑动平均后loss，清空累积loss
    def update_and_get_ema_losses(self):
        res=[]
        self.total_weighted_loss=0
        for i in range(len(self.EMA_list)):
            now=self.EMA_list[i].update(self.accum_loss_list[i])#每个loss进行滑动平均
            res.append(now)
            self.total_weighted_loss+=(now*self.weight_list[i])#加权求和
        self.accum_loss_list=[0]*len(self.weight_list)#清空累积
        return res,self.total_weighted_loss

def baseTokenize(tk:BertTokenizer,inp:str):#做最简单的事，str分词为list[str]
    if inp is None:
        return inp
    inp='_'.join(inp.split())#空格等替换成下划线字符，防止被模型忽略
    inp=inp.translate(NORMAL_TOKENS).lower()
    tokens=jieba.lcut(inp,HMM=False)
    res=[]
    for i in tokens:
        if i in tk.vocab:
            res.append(i)
        else:
            if i.isascii():#英文单词分成一个个字符
                res+=list(i)
            else:
                res+=tk.tokenize(i)
    return res

def ChineseTokenize(tk:BertTokenizer,text:str,textPair:str=None,
                    returnIds=False,add_special_tokens=True,maxlen=None):

    maxlen=9999999 if maxlen is None else maxlen
    if add_special_tokens:
        maxlen-=2 if textPair is None else 3#特殊字符占用
    assert maxlen>=0
    b=[]
    if textPair is None:#根据是否有第二段进行不同处理
        a=baseTokenize(tk,text)[:maxlen]
    else:
        a,b=truncate_pair(baseTokenize(tk,text),baseTokenize(tk,textPair),maxlen)
    if add_special_tokens:#看要不要加特殊字符
        a=[tk.cls_token]+a+[tk.sep_token]
        b=b+[tk.sep_token]
    input_ids=a if textPair is None else a+b
    if returnIds:
        input_ids=tk.convert_tokens_to_ids(input_ids)
    seg=[0]*len(a)+([] if textPair is None else [1]*len(b))
    return {'input_ids':input_ids,'token_type_ids':seg}


def decodeText(textIds,tk):
    #如果不skip_special_tokens，除了不在词表的unk外，还会有batch时的pad
    res=tk.convert_ids_to_tokens(textIds,skip_special_tokens=True)
    #print(res)
    res="".join(res).replace("##", "").strip()
    return res

#预测阶段用
class seq2seqDataForPre(Dataset):
    #传入句子对列表
    def __init__(self,textLs:list,tk:BertTokenizer):
        super().__init__()
        self.tk=tk
        self.origin_data=textLs
        self.aug_data=[]
        for now in self.origin_data:
            a=now.get("answer",[])
            input_ids,token_type_ids=encodeText(now,self.tk,shuffle_cd=False)
            self.aug_data.append({'input_ids':input_ids,'token_type_ids':token_type_ids,'answer':"".join(a),'id':now.get('id',0)})
        self.aug_data.sort(key=lambda x:len(x['input_ids']),reverse=True)

    def __len__(self):
        return len(self.origin_data)

    #耗时操作在此进行，可用上多进程
    def __getitem__(self, item):
        now=dict(self.aug_data[item])
        for k in ['input_ids']:
            now[k]=self.tk.convert_tokens_to_ids(now[k])
        return now

    @classmethod
    def collate(cls,batch):
        now=dict()
        for k,padding_v in zip(['input_ids','token_type_ids'],[0,0]):
            v=[i[k] for i in batch]
            v=paddingList(v,padding_v,returnTensor=True)
            now[k]=v
        now['attention_mask']=(now['input_ids']!=0)
        return now


def cal_bleu12_f1(true_ls,pred_ls):
    assert len(true_ls)==len(pred_ls)
    true=[list(i) for i in true_ls]
    pred=[list(i) for i in pred_ls]
    #bleu1
    score=[bleu([i], j,weights=[1]) for i,j in zip(true,pred)]
    #print(score)
    bleu1=sum(score)/len(score)
    #blue2
    score=[bleu([i], j,weights=[0,1]) for i,j in zip(true,pred)]
    #print(score)
    blue2=sum(score)/len(score)
    #rouge1_f
    pred=[" ".join(i) for i in pred_ls]#基于字
    true=[" ".join(i) for i in true_ls]
    rouge = Rouge(exclusive=False,metrics=["rouge-1"])
    rg=rouge.get_scores(pred,true,avg=True,ignore_empty=True)
    #print(rg)
    rouge1_f=rg['rouge-1']['f']
    return bleu1,blue2,rouge1_f

def cal_knowledge_selection(true_ls,pred_ls):
    def cal_one(gold_knowledge, pred_knowledge):
        gold_knowledge=[] if gold_knowledge is None else gold_knowledge
        pred_knowledge=[] if pred_knowledge is None else pred_knowledge
        if len(gold_knowledge) == 0 and len(pred_knowledge) == 0:
            return 1.0, 1.0, 1.0
        gold_knowledge, pred_knowledge=set(tuple(_) for _ in gold_knowledge),set(tuple(_) for _ in pred_knowledge)
        precision, recall, f1 = 0.0, 0.0, 0.0
        relevance = gold_knowledge&pred_knowledge
        if len(relevance) == 0 or len(relevance) == 0:
            return precision, recall, f1
        tp = len(relevance)
        precision = tp / len(pred_knowledge) if len(
            pred_knowledge) > 0 else 0.0
        recall = tp / len(gold_knowledge) if len(gold_knowledge) > 0 else 0.0
        if precision == 0 and recall == 0:
            return precision, recall, f1
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1
    true_ls=[[] if _ is None else _ for _ in true_ls]
    pred_ls=[[] if _ is None else _ for _ in pred_ls]
    res=[cal_one(true,pred) for true,pred in zip(true_ls,pred_ls)]
    Len=len(true_ls)
    precision, recall, f1=sum(_[0] for _ in res)/Len,sum(_[1] for _ in res)/Len,sum(_[2] for _ in res)/Len
    return precision, recall, f1

