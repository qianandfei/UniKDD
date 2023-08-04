# -*- coding: utf-8 -*-
# @Time    : 2022/9/1 11:03
# @Author  : zhangweiqi
#! -*- coding: utf-8 -*-

#====修改的进行替换====
import transformers
from edited_transformers import generation_utils
transformers.generation_utils.GenerationMixin=generation_utils.GenerationMixin
from modeling_T5_KG import T5_KG_Model
from transformers.optimization import AdamW,get_polynomial_decay_schedule_with_warmup
from transformers import BertTokenizer
from Config import Config
from Trainer import Trainer
from BertUtils import seq2seqData,blockShuffleDataLoader,seq2seqDataForPre,cache_data,encodeText,cache_kg
from torch.utils.data import DataLoader
from NLP_Utils import readFromJsonFile,readFromJsonFileForLine
from data_prepare import prepare_data
import traceback
import jieba
import logging
jieba.setLogLevel(logging.INFO)
#====随机种子====
import random
import torch
torch.set_num_threads(1)#若不设，不知为何自动占用了大量cpu核，消耗大量资源还降低了速度
import numpy as np
import os
seed=0
#python和numpy
random.seed(seed)
os.environ['PYTHONHASHSEED'] =str(seed)#消除hash算法的随机性
np.random.seed(seed)
#torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)#当前使用gpu
torch.backends.cudnn.benchmark=False#卷积相关，关了会变慢，但保证卷积可复现
torch.backends.cudnn.deterministic=True

prepare_data()#改造数据集#预处理候选  生成final訓練接 kg全域知识图谱 pred 最后预测结果文件 val 改造好的验证集 test 改造好测试集 （val,test都是滑动窗口写的）
cache_kg()#改变知识图谱格式并保存在内存中
tk=BertTokenizer.from_pretrained(Config.pretrained_path)
train_data=cache_data(tk)#缓存分词后的数据 加速训练 train_cache.josn 对于训练集
train_data=seq2seqData(train_data,tk)# 为dataloader 准备数据集
train_dataLoader=blockShuffleDataLoader(train_data,len(train_data)//Config.batch_size//5,shuffle=False
                                        ,batch_size=Config.batch_size,collate_fn=seq2seqData.collate)
val_data=readFromJsonFile(Config.val_data_path)
model=T5_KG_Model(Config.pretrained_path).to(Config.device)

# history_ls,true_triplet_ls,true_response_ls=[_[0] for _ in val_data],[_[1] for _ in val_data],[_[2] for _ in val_data]
# metrics,gen_res=model.gen_val_batch(history_ls,true_triplet_ls,true_response_ls,tk,disable_tqdm=False)

optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in Config.no_decay)], 'weight_decay': Config.weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in Config.no_decay)], 'weight_decay': 0.0}
]
opt=AdamW(optimizer_grouped_parameters,lr=Config.lr_begin)

scheduler=get_polynomial_decay_schedule_with_warmup(
            opt, num_warmup_steps=Config.warmup_step, num_training_steps=Config.epoch*len(train_dataLoader),lr_end=Config.lr_end
        )
trainer=Trainer(model,opt,train_dataLoader,tk,scheduler,val_data)


if __name__ == '__main__':
    try:
        trainer.train()
    except:
        trainer.save_model(Config.save_path+'interrupted/',Config.global_step/len(trainer.train_dataLoader),trainer.losses.total_weighted_loss,0,True)
        traceback.print_exc()#打印异常
        # trainer.model.load_state_dict(Config.bestWeights)#最佳模型也保存
        # trainer.save_model(Config.save_path+'best/',Config.last_better_step/len(trainer.train_dataLoader),Config.best_cp_loss,Config.best_val_score)





