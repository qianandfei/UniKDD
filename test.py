# -*- coding: utf-8 -*-
# @Time    : 2022/9/16 17:43
# @Author  : zhangweiqi
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
from NLP_Utils import readFromJsonFile,readFromJsonFileForLine,writeToJsonFile
import traceback
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

# final_path=os.listdir(f"{Config.save_path}last/")[0]
Config.pretrained_path=f"{Config.save_path}epoch-9.0-loss-5.572e-01-metrics-1.334e+00/"#f"{Config.save_path}last/{final_path}/"
#Config.pretrained_path='save/last/epoch-11.0-loss-3.380e-01-metrics-0.000e+00/'
print('模型路径：',Config.pretrained_path)
Config.num_beams=5
Config.batch_size=5
model=T5_KG_Model(Config.pretrained_path).to(Config.device)
cache_kg()
tk=BertTokenizer.from_pretrained(Config.pretrained_path)

#验测试集
test_data=readFromJsonFile(Config.test_data_path)

history_ls,true_triplet_ls,true_response_ls=[_[0] for _ in test_data],[_[1] for _ in test_data],[_[2] for _ in test_data]
metrics,gen_res=model.gen_val_batch(history_ls,true_triplet_ls,true_response_ls,tk,disable_tqdm=False)
writeToJsonFile('data/pred_test.json',gen_res,indent=4)
# ans={}
# for one_test,one_res in zip(test_data,gen_res):
#     pred_triplet=one_res["pred_triplet"]
#     pred_response=one_res["pred_response"]
#     ans[one_test[0]]=dict()
#     if pred_triplet is not None and len(pred_triplet)>0:
#         pred_triplet=[{"attrname": _[1],"attrvalue": _[2],"name": _[0]} for _ in pred_triplet]
#         ans[one_test[0]]["attrs"]=pred_triplet
#     ans[one_test[0]]["message"]=pred_response
#
# writeToJsonFile(Config.result_path,ans,indent=4)







