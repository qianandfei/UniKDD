# -*- coding: utf-8 -*-
# @Time    : 2022/9/1 10:28
# @Author  : zhangweiqi
#! -*- coding: utf-8 -*-

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
from collections import OrderedDict
from BertUtils import encodeText, decodeText,ChineseTokenize,baseTokenize,paddingList,splitList,encodeEntityPred,\
    encodeRelPred,union_triplet,match_entity,NORMAL_TOKENS,cal_bleu12_f1,cal_knowledge_selection,match_entity_ls
from Config import Config
from edited_transformers.modeling_t5 import T5ForConditionalGeneration, T5Config

class T5_KG_Model(torch.nn.Module):
    def __init__(self,pretrained_path:str):
        super().__init__()
        self.cfg=T5Config.from_pretrained(pretrained_path)
        self.generative_model:T5ForConditionalGeneration=T5ForConditionalGeneration(self.cfg)
        self.loss_for_gen=CrossEntropyLoss(ignore_index=0)#忽略padding部分
        state_dict=torch.load(pretrained_path+'pytorch_model.bin',map_location=torch.device('cpu'))
        if 'generative_model.shared.weight' not in state_dict:#最初训练时从T5加载大部分权重
            print("仅加载T5权重：\n",self.generative_model.load_state_dict(state_dict,strict=False))
        else:#之后直接加载所有
            print("加载所有权重：\n",self.load_state_dict(state_dict,strict=False))

    def gen_cal_loss(self,data):
        inputs=dict()
        inputs['input_ids'],inputs['decoder_input_ids']=data
        inputs['attention_mask']=(inputs['input_ids']!=0)#自动加attention mask
        inputs['decoder_attention_mask']=(inputs['decoder_input_ids']!=0)
        outputs=self.generative_model(output_hidden_states=True,**inputs)
        lm_logits=outputs['logits'][:, :-1].contiguous()
        lm_logits=lm_logits.reshape(-1,lm_logits.shape[-1])
        label=inputs['decoder_input_ids'][:,1:].reshape(-1)
        loss=self.loss_for_gen(lm_logits,label)
        return loss,outputs['decoder_hidden_states']#把编码器隐状态也返回便于计算rank模型损失


    #基本的解码，给定batch输入，batch解码输入
    #inputs为已经token化后的输入列表
    @torch.no_grad()  #接受已经处理好的输入（加好控制吗等） 直接返回输出
    def gen_batch(self,inputs:list,tk:BertTokenizer,decoder_start_token='[gen_response]',decoder_prompt=None,disable_tqdm=True):
        if self.training:
            self.eval()
        decoder_prompt=[] if decoder_prompt is None else decoder_prompt
        decoder_prompt=tk.convert_tokens_to_ids(decoder_prompt)

        res=[list(_) for _ in enumerate(inputs)]
        res.sort(key=lambda x:len(x[1]),reverse=True)
        #print("\n开始验证生成模型……")
        #设置总长，解码部分最长为max_dec_len（加2是因为起始符cls和结束符sep），且总体不要超过Config.maxlen
        maxLen=min(Config.max_dec_len,Config.dec_maxlen)
        minLen=min(Config.min_dec_len,Config.dec_maxlen)
        bs,Len=Config.batch_size*2,len(res)
        for be in tqdm(range(0,Len,bs),ncols=80,disable=disable_tqdm):
            now_input=res[be:be+bs]
            input_ids=[tk.convert_tokens_to_ids(i[1]) for i in now_input]
            input_ids=paddingList(input_ids,0,True)
            attention_mask=(input_ids!=0)
            #generate函数会自动关梯度
            now=self.generative_model.generate(input_ids=input_ids,decoder_start_token_id=tk.convert_tokens_to_ids(decoder_start_token),eos_token_id=tk.sep_token_id,use_cache=True,
                                    attention_mask=attention_mask,decoder_prompt=decoder_prompt,#decoder_start_token
                                    max_length=maxLen,min_length=minLen,#decoder_start_token_id =控制码
                                    num_beams=Config.num_beams,#       #decoder_prompt--
                                    do_sample=Config.do_sample,top_k=Config.sample_topk,num_return_sequences=Config.num_return_sequences,)
            now=[decodeText(i[1:-1],tk) for i in now]#如果num_return_sequences>1，每条样本返回的多条会展平在一起
            if Config.num_return_sequences>1:
                now=splitList(now,Config.num_return_sequences)
            for a,b in zip(now_input,now):
                a[1]=b
        res.sort(key=lambda x:x[0])
        res=[_[1] for _ in res]
        return res


    #解码实体
    #history为对话历史，batch输入
    @torch.no_grad()
    def gen_entity_batch(self,history_ls:list,tk:BertTokenizer,disable_tqdm=True):
        if self.training:
            self.eval()
        Config.do_sample=False#需确定性解码
        input_ids=[encodeEntityPred(_,None,tk)[0] for _ in history_ls]
        res=self.gen_batch(input_ids,tk,decoder_start_token='[gen_entity]',disable_tqdm=disable_tqdm)
        return res

    #解码关系
    #history为对话历史，entity为每个history对应的实体，batch输入
    @torch.no_grad()
    def gen_rel_batch(self,history_ls:list,entity_ls:list,tk:BertTokenizer,disable_tqdm=True):
        if self.training:
            self.eval()
        Config.do_sample=False#需确定性解码
        input_ids=[]
        for his,ent in zip(history_ls,entity_ls):
            e_r_dict={ent:Config.all_kg[ent][1]}
            input_ids.append(encodeRelPred(his,[ent,[]],e_r_dict,tk)[0])
        res=self.gen_batch(input_ids,tk,decoder_start_token='[gen_relation]',disable_tqdm=disable_tqdm)
        return res

    #解码对话
    #history为对话历史，kg_ls为每个history对应的文本化后的三元组知识列表，batch输入
    @torch.no_grad()
    def gen_response_batch(self,history_ls:list,kg_ls:list,tk:BertTokenizer,disable_tqdm=True):
        if self.training:
            self.eval()
        Config.do_sample=False#自动评价时response不要随机
        input_ids=[]
        for his,kg in zip(history_ls,kg_ls):
            kg=[baseTokenize(tk,_) for _ in kg]
            now=encodeText(his,kg,tk)
            input_ids.append(now)
        res=self.gen_batch(input_ids,tk,decoder_start_token='[gen_response]',disable_tqdm=disable_tqdm)
        res=[i.split('[bot]')[-1] for i in res]
        return res

    #全流程解码，输入history和真实response（可选），自动根据history获取知识预测response，batch输入
    #[(history, response, true_triplet)]，默认输入未分词
    #真实response可能为空
    @torch.no_grad()
    def gen_val_batch(self,history_ls:list,true_triplet_ls:list,true_response_ls:list,tk:BertTokenizer,cal_metrics=True,disable_tqdm=True):
        if self.training:
            self.eval()
        history_bak=history_ls
        history_ls=[[baseTokenize(tk,_) for _ in history] for history in history_ls]
        #先生成实体
        entity_ls=self.gen_entity_batch(history_ls,tk,disable_tqdm=disable_tqdm)
        #print('抽取实体：',entity_ls)
        entity_ls=match_entity_ls(entity_ls,disable_tqdm=disable_tqdm)
        #print('实体链接后：',entity_ls)
        res,temp=[],[]

        for id,(history,entity) in enumerate(zip(history_ls,entity_ls)):
            res.append({
                'history':history,
                'true_triplet':[],
                'true_response':None,
                'pred_response':None,
                'pred_entity':entity,
                'pred_rel':None,#预测的实体关系先用None占位
                'pred_triplet':[],
                'pred_kg':None,

            })
            #把生成非空实体的取出来，进行关系预测
            if entity!='无':
                temp.append([id,history,entity])
        #对需要进行知识预测的，预测对应关系和三元组
        id_ls,history_ls,entity_ls=[_[0] for _ in temp],[_[1] for _ in temp],[_[2] for _ in temp]
        rel_ls=self.gen_rel_batch(history_ls,entity_ls,tk,disable_tqdm=disable_tqdm)
        #print('预测关系：',rel_ls)
        triplet_ls,kg_ls=[],[]
        for entity,rel in zip(entity_ls,rel_ls):
            rel=set(rel.split('/'))
            triplet=[]
            for one in Config.all_kg[entity][2]:
                rel_temp=one[1]
                rel_temp='_'.join(rel_temp.split()).translate(NORMAL_TOKENS).lower()
                if rel_temp in rel:
                    triplet.append(tuple(one))
            triplet=list(OrderedDict([(_,None) for _ in triplet]).keys())#去重一下
            triplet_ls.append(triplet)
            kg_ls.append(union_triplet(triplet))
        #写回
        for idx,rel,triplet,kg in zip(id_ls,rel_ls,triplet_ls,kg_ls):
            res[idx]['pred_rel']=rel
            res[idx]['pred_triplet']=triplet
            res[idx]['pred_kg']=kg
        #进行最后的response预测
        history_ls=[_['history'] for _ in res]
        pred_kg_ls=[[] if _['pred_kg'] is None else _['pred_kg'] for _ in res]
        pred_response_ls=self.gen_response_batch(history_ls,pred_kg_ls,tk,disable_tqdm=disable_tqdm)
        #pred_response_ls=[_.replace('_',' ') for _ in pred_response_ls]
        #response写回
        for one,response,history in zip(res,pred_response_ls,history_bak):
            one['history']=history
            one['pred_response']=response

        metrics=-1
        #指标计算
        if cal_metrics and len(true_response_ls)!=None:
            pred_triplet_ls=[[] if _['pred_triplet'] is None else _['pred_triplet'] for _ in res]
            precision,recall,f1=cal_knowledge_selection(true_triplet_ls,pred_triplet_ls)
            print(f"\n知识选择模型当前指标 -> precision：{precision}，recall：{recall}，f1：{f1}")

            bleu1,blue2,rouge1_f=cal_bleu12_f1(true_response_ls,pred_response_ls)#返回指标、预测结果
            print(f"\n生成模型当前指标 -> bleu1：{bleu1}，blue2：{blue2}，rouge1_f：{rouge1_f}")
            for one,response,triplet in zip(res,true_response_ls,true_triplet_ls):
                one['true_response']=response
                one['true_triplet']=triplet
            metrics=0.3*(precision+recall+f1)+0.7*(bleu1+blue2+rouge1_f)
        print(f"\n模型总分数：{metrics}")
        return metrics,res






