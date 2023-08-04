# -*- coding: utf-8 -*-
# @Time    : 2022/9/1 11:37
# @Author  : zhangweiqi
#! -*- coding: utf-8 -*-

from Config import Config
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from BertUtils import Multi_accum_ema_loss
from transformers import BertTokenizer
import os
import shutil
from modeling_T5_KG import T5_KG_Model
from torch.cuda.amp import autocast,GradScaler
from contextlib import  nullcontext
from NLP_Utils import writeToJsonFile
scaler = GradScaler()
import sys
sys.setrecursionlimit(5000000)#设置最大递归深度，计算LCS需要

import torch
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.25, emb_name='shared'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='shared'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class Trainer():
    def __init__(self,model:T5_KG_Model,#模型
                 opt:torch.optim.Optimizer,#优化器
                 train_dataLoader:DataLoader,#训练数据
                 tk:BertTokenizer,#分词器
                 scheduler:torch.optim.lr_scheduler.LambdaLR,
                 val_data,#验证数据
                 ):
        self.model=model
        self.fgm = FGM(model)
        self.opt=opt
        self.train_dataLoader=train_dataLoader
        self.tk=tk
        self.val_data=val_data
        self.losses=Multi_accum_ema_loss(list(Config.loss_weights.values()))
        self.scheduler=scheduler


    def train_one_step(self,inputs):
        contex=autocast if Config.fp16 else nullcontext
        with contex():
            gen_loss,decoder_hidden_states=self.model.gen_cal_loss(inputs) if Config.loss_weights['gen']!=0 else (torch.tensor(0),None)#计算生成损失
            #cl_loss=self.model.match_cal_loss(inputs) if Config.loss_weights['cl']!=0 else torch.tensor(0)
            cl_loss=torch.tensor(0)
            self.losses.add_accum_loss([gen_loss.item()/Config.grad_accum_steps,cl_loss.item()/Config.grad_accum_steps])
            loss=gen_loss*Config.loss_weights['gen']+cl_loss*Config.loss_weights['cl']
            loss/=Config.grad_accum_steps#loss需要在累积步间平均
        scaler.scale(loss).backward() if Config.fp16 else loss.backward()#反向传播
        # if Config.fgm_e!=0:
        #     self.fgm.attack(epsilon=Config.fgm_e)
        #     with contex():
        #         gen_loss,decoder_hidden_states=self.model.gen_cal_loss(inputs) if Config.loss_weights['gen']!=0 else (torch.tensor(0),None)#计算生成损失
        #         cl_loss=self.model.cl_cal_loss(decoder_hidden_states) if Config.loss_weights['cl']!=0 else torch.tensor(0)
        #         #self.losses.add_accum_loss([gen_loss.item()/Config.grad_accum_steps,cl_loss.item()/Config.grad_accum_steps])
        #         loss=gen_loss*Config.loss_weights['gen']+cl_loss*Config.loss_weights['cl']
        #         loss/=Config.grad_accum_steps#loss需要在累积步间平均
        #     scaler.scale(loss).backward() if Config.fp16 else loss.backward()#反向传播
        #     self.fgm.restore()

    #超长的batch，分割成多次训练防止oom
    def adaptive_bs_train_one_step(self,inputs):
        bs,Len=inputs[0].shape[0],inputs[0].shape[1]
        if Len<=300:
            self.train_one_step(inputs)
        else:#超长的，统一分割成bs计算
            if 300<Len<=500:
                small_bs=16
            else:
                small_bs=8
            for be in range(0,bs,small_bs):
                now=[_[be:be+small_bs] for _ in inputs]
                self.train_one_step(now)

    #训练
    def train(self):
        self.model.train()
        for epoch in range(Config.epoch):
            epoch_iterator=tqdm(self.train_dataLoader,ncols=140,mininterval=0.3)#长度与显示频率
            for inputs in epoch_iterator:
                Config.global_step+=1
                if not self.model.training:#保证在training模式
                    self.model.train()
                self.adaptive_bs_train_one_step(inputs)#超长的batch，分割成多次训练然后梯度累积  防止oom
                if Config.global_step%Config.grad_accum_steps==0:#需要更新了
                    (scaler.step(self.opt),scaler.update()) if Config.fp16 else self.opt.step()
                    self.opt.zero_grad()#清空梯度
                    (gen_show_loss,cl_show_loss),weighted_show_loss=self.losses.update_and_get_ema_losses()
                    epoch_iterator.set_description_str(f"epoch：{epoch+1}",refresh=False)
                    epoch_iterator.set_postfix_str("weighted_loss：{:.3e}，gen_loss：{:.3e}，cl_loss：{:.3e}，lr：{:.3e}".format(
                        weighted_show_loss,gen_show_loss,cl_show_loss,self.scheduler.get_last_lr()[0]),refresh=False)
                self.scheduler.step()#更新学习率
                #验证
                if self.val_data is not None and Config.global_step%Config.val_step==0 and epoch+1>=Config.start_val_epoch and self.losses.total_weighted_loss<=Config.start_val_loss:#进行验证
                    print()
                    history_ls,true_triplet_ls,true_response_ls=[_[0] for _ in self.val_data],[_[1] for _ in self.val_data],[_[2] for _ in self.val_data]
                    metrics,gen_res=self.model.gen_val_batch(history_ls,true_triplet_ls,true_response_ls,self.tk,disable_tqdm=False)
                    if metrics>Config.best_val_score:
                        Config.best_val_score=metrics
                        Config.last_better_step=Config.global_step
                        Config.bestWeights={k:v.cpu().clone() for k, v in self.model.state_dict().items()}#保存最佳权重
                        Config.best_cp_loss=self.losses.total_weighted_loss
                        print(f"\n找到更佳模型，当前得分：{Config.best_val_score}\n")
                        writeToJsonFile('data/pred.json',gen_res,indent=4)
                    else:
                        print(f"\n未改进，当前得分：{metrics}，最佳得分：{Config.best_val_score}\n")
                        if Config.global_step-Config.last_better_step>=Config.early_stop_step:
                            self.save_model(Config.save_path+'last/',Config.global_step/len(self.train_dataLoader),self.losses.total_weighted_loss,metrics,clear_path=True)#最后模型备份一下
                            print(f"\n长时间未改进，提前停止训练\n")
                            self.model.load_state_dict(Config.bestWeights)#先加载最佳权重
                            self.save_model(Config.save_path,Config.last_better_step/len(self.train_dataLoader),Config.best_cp_loss,Config.best_val_score)
                            return
                #备份
                if Config.global_step%Config.bak_step==0:
                    self.save_model(Config.save_path+'bak/',Config.global_step/len(self.train_dataLoader),self.losses.total_weighted_loss,0,clear_path=False)
                    print("\n备份成功\n")
            epoch_iterator.close()
        #能训练到终止
        self.save_model(Config.save_path+'last/',Config.global_step/len(self.train_dataLoader),self.losses.total_weighted_loss,0,clear_path=True)#最后模型备份一下
        print(f"\n训练完成\n")
        # self.model.load_state_dict(Config.bestWeights)#最佳模型也保存
        # self.save_model(Config.save_path+'best/',Config.last_better_step/len(self.train_dataLoader),Config.best_cp_loss,Config.best_val_score)


    def save_model(self,path,epoch,loss,metrics,clear_path=False):#保存模型的文件夹附带loss等信息，clear_path是否先清空该目录
        if clear_path and os.path.exists(path):
            shutil.rmtree(path)
            os.makedirs(path)
        savePath=path+f"epoch-{epoch:.1f}-loss-{loss:.3e}-metrics-{metrics:.3e}/"
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        torch.save(self.model.state_dict(),savePath+'pytorch_model.bin')
        self.model.generative_model.config.save_pretrained(savePath)
        self.tk.save_vocabulary(savePath)

