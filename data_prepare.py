# -*- coding: utf-8 -*-
# @Time    : 2022/9/22 16:28
# @Author  : zhangweiqi
from collections import OrderedDict

from tqdm import tqdm
import os
from BertUtils import NORMAL_TOKENS
from BertUtils import union_triplet
from Config import Config
from NLP_Utils import readFromJsonFile, writeToJsonFile

def simplify_kdconv(kg):
    temp=[(i['name'],i['attrname'],str(i['attrvalue'])) for i in kg]
    temp=list(OrderedDict([(_,None) for _ in temp]).keys())
    return temp


def prepare_kg(all_kg:dict,all_data:list):
    print('删除歧义前数据库实体数：',len(all_kg))
    all_entity=set()
    for now in all_data:
        for idx,text in enumerate(now['messages']):
            tripet=simplify_kdconv(text.get('attrs',[]))
            all_entity.update([_[0] for _ in tripet])
    deleted=set()
    for k,v in all_kg.items():
        if '（' in k or '(' in k:
            no_k=k.replace('(','（').split('（')[0]
            if no_k in all_kg:#存在有无括号的歧义
                if k in all_entity and no_k in all_entity:#在训练集中也存在歧义的
                    pass
                elif k in all_entity and no_k not in all_entity:
                    deleted.add(no_k)
                    pass
                elif k not in all_entity and no_k in all_entity:
                    deleted.add(k)
                    pass
    for i in deleted:
        assert i not in all_entity
        all_kg.pop(i)
    deleted=set()
    for k,v in all_kg.items():
        if '（' in k or '(' in k:
            no_k=k.replace('(','（').split('（')[0]
            if no_k in all_kg and (no_k not in all_entity):#存在有无括号的歧义
                deleted.add(no_k)
    for i in deleted:
        assert i not in all_entity
        all_kg.pop(i)
    print('删除歧义后数据库实体数：',len(all_kg))
    return all_kg

def get_e_r(kg):
    ans=OrderedDict([(e,set()) for e,r,p in kg])
    for e,r,p in kg:
        ans[e].add(r)
    return list(ans.items())

def normalize(data,all_kg):
    temp=[]
    for i in tqdm(range(len(data)),ncols=80):
        ut=[]
        e_r=dict()
        for idx,text in enumerate(data[i]['messages']):
            tripet=simplify_kdconv(text.get('attrs',[]))
            kg=union_triplet(tripet)
            tripet=get_e_r(tripet)
            text=text['message']
            for idx_t,(entity,rel) in enumerate(tripet):
                if entity not in e_r:
                    e_r[entity]=set()
                e_r[entity]=e_r[entity]|rel
                #字典转列表防止无法格式化
                tripet[idx_t]=(entity,list(rel))
            if idx==0:
                ut.append([text,True,[],[]])
            else:
                ut.append([text,False,kg,tripet])
        #从数据库中更新所有关系
        for entity,rel in e_r.items():
            e_r[entity]=list(e_r[entity]|all_kg[entity])
        now={'kg':[[],[],[]],'e_r':e_r,'ut':ut}
        temp.append(now)
    return temp


def prepare_val(data):#滑窗产生验证集和测试集
    temp=[]
    for i in tqdm(range(len(data)),ncols=80):
        ut=[_["message"] for _ in data[i]['messages']]
        for idx,text in enumerate(data[i]['messages']):
            if idx==0:
                continue
            tripet=simplify_kdconv(text.get('attrs',[]))
            temp.append([
                ut[:idx],#历史对话
                tripet,#知识
                ut[idx]#回复
            ])
    return temp

def prepare_data():
    if os.path.exists(Config.train_data_path):
        return
    print('开始准备数据集……')
    #读入两个数据集
    path=Config.kdconv_path
    kdconv=[]
    for j in ['train','dev','test']:
        for i in ['film','music','travel']:
            now=path+i+'/'+j+'.json'
            kdconv+=readFromJsonFile(now)
        print(len(kdconv))

    val,test=prepare_val(kdconv[3600:4050]),prepare_val(kdconv[4050:])
    print(f'滑窗后验证集大小：{len(val)}，测试集大小：{len(test)}')
    #滑窗得到验证集和测试集
    writeToJsonFile(Config.val_data_path,val,indent=4)
    writeToJsonFile(Config.test_data_path,test,indent=4)
    #读入数据库
    if os.path.exists(Config.kg_path):
        all_kg=readFromJsonFile(Config.kg_path)
    else:#先删除带歧义的
        path=Config.kdconv_path
        all_kg=dict()
        for i in ['film','music','travel']:
            now=path+i+'/'+f'kb_{i}.json'
            all_kg.update(readFromJsonFile(now))
            print(f'数据库大小：{len(all_kg)}')
        writeToJsonFile(Config.kg_path,all_kg,indent=4)
    #%%
    kg_num=[]
    rel_num=[]
    rel_len=[]
    relation=set()
    #改造成{实体：{所有属性}}
    for k,v in all_kg.items():
        kg_num.append(len(v))
        now_r={_[1] for _ in v}
        all_kg[k]=now_r
        relation.update(now_r)
        rel_num.append(len(now_r))
        now_r='_'.join(now_r)
        rel_len.append(len(now_r))
    len(relation)
    #%%
    #先转换格式
    #得到[(该句所用实体，[该句该实体所有用了的属性]), ]
    train=normalize(kdconv[:3600],all_kg)
    #%%
    final={'kdconv':train}
    for k,v in final.items():
        print(k,len(v))
    writeToJsonFile(Config.train_data_path,final,indent=4)


def simplify_model_vocab(modelPath,newModelPath):# 删除多余unused占位符 同时推字表和模型嵌入的一起操作
    """
    modelPath：旧模型目录，包括config、词表、模型文件
    newModelPath：这个目录最初只需放置新的词表，修改后的模型、config文件将保存到此目录
    """
    import json,torch
    from transformers import MT5ForConditionalGeneration,MT5Config,BertTokenizer
    def writeToJsonFile(path: str, obj,indent=4):
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False,indent=indent,sort_keys=True))
    def readFromJsonFile(path: str):
        with open(path, "r", encoding="utf-8") as f:
            return json.loads(f.read())
    def saveVocab(path:str,obj,sortVocab=False):
        obj=list(obj)
        if sortVocab:
            obj.sort()
        with open(path, "w", encoding="utf-8") as f:
            for i in obj:
                f.write(i+'\n')
    bert=MT5ForConditionalGeneration.from_pretrained(modelPath)
    bert.eval()
    oldTk=BertTokenizer.from_pretrained(modelPath)
    newTk=BertTokenizer.from_pretrained(newModelPath)
    #%%

    #%%
    ### 一、记录找得到的，和找不到只能随机或平均初始化的token，生成新config文件
    #%%
    deled,added=list(oldTk.vocab.keys()-newTk.vocab.keys()),list(newTk.vocab.keys()-oldTk.vocab.keys())
    deled.sort(),added.sort()
    #记录新旧vocab的token变化
    print(f"原vocab删除{len(deled)}个，新vocab增加{len(added)}个")
    saveVocab(newModelPath+'原vocab已删除.json',deled,True)
    saveVocab(newModelPath+'新vocab已增加.json',added,True)
    print(f"删除与新增token、新的config已写入路径：{newModelPath}",)
    oldConfig=readFromJsonFile(modelPath+'config.json')
    oldConfig['vocab_size']=newTk.vocab_size
    oldConfig['is_decoder']=True
    #写入新的config文件，变化的也就vocab大小
    writeToJsonFile(newModelPath+'config.json',oldConfig)
    #%% md
    ### 二、共有的token直接复制权重
    #%%
    common=oldTk.vocab.keys()&newTk.vocab.keys()
    #%%
    #要改shared.weight
    #lm_head.weight
    #%%
    bert.lm_head.weight
    #%%
    #对于旧词表中能找到的，记录下每个key的embedding和mlm bias的权重
    key2embedding,key2mlmWeight=dict(),dict()
    for key in common:
        idx=oldTk.vocab[key]
        val=bert.shared.weight.data[idx]
        key2embedding[key]=val
        val=bert.lm_head.weight.data[idx]
        key2mlmWeight[key]=val
    #%% md
    ### 三、旧词表模型中找不到的token，尝试用旧词表细粒度拆分后取平均权重
    #%%
    #%%
    for key in added:
        idx=oldTk.encode(key,add_special_tokens=False)#切分后还找不到的就变成了unk
        val=bert.shared.weight.data[idx].mean(dim=0)
        key2embedding[key]=val
        val=bert.lm_head.weight.data[idx].mean(dim=0)
        key2mlmWeight[key]=val
    #%% md
    ### 四、开始恢复和保存
    #%%
    #新模型调整下embedding层，mlm层大小
    bert.resize_token_embeddings(newTk.vocab_size)
    #%%
    #根据dict恢复权重
    for key in newTk.vocab.keys():
        idx=newTk.vocab[key]#找到在新模型里的index
        val=key2embedding[key]
        bert.shared.weight.data[idx]=val
        val=key2mlmWeight[key]
        bert.lm_head.weight.data[idx]=val
    #%%
    #保存新模型
    torch.save(bert.state_dict(),newModelPath+'pytorch_model.bin')
    #%%
    ### 五、检查权重是否符合预期
    #%%
    old=torch.load(open(modelPath+'pytorch_model.bin',"rb"))
    new=torch.load(open(newModelPath+'pytorch_model.bin',"rb"))
    #%%
    for key in common:
        oldIdx=oldTk.vocab[key]
        newIdx=newTk.vocab[key]
        notEqual=old['shared.weight'][oldIdx]!=new['shared.weight'][newIdx]
        assert notEqual.sum().item()==0
    for key in common:
        oldIdx=oldTk.vocab[key]
        newIdx=newTk.vocab[key]
        notEqual=old['lm_head.weight'][oldIdx]!=new['lm_head.weight'][newIdx]
        assert notEqual.sum().item()==0


if __name__ == '__main__':
    prepare_data()








