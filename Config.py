# -*- coding: utf-8 -*-
# @Time    : 2022/1/15 23:25
# @Author  : zhangweiqi
#! -*- coding: utf-8 -*-


#训练调参专用
class Config:
    #数据相关
    pretrained_path='save/t5_base/' #r'model/t5_small/'
    save_path='save/'
    train_data_path='data/final.json'#[:3600],[3600:4050],[4050:]
    train_cache_path='data/train_cache.json'
    kg_path='data/kg.json'
    val_data_path='data/val.json'
    test_data_path='data/test.json'
    kdconv_path='data/kdconv/'
    result_path='data/result.json'


    #训练相关
    fuzzy_prob=0.0
    device=['cuda','cpu'][0]
    fgm_e=0
    fp16=True
    blockShuffle_gen_data=False#是否分块shuffle生成数据以加速训练，开启后可能将会降低性能
    loss_weights={"gen":1,"cl":0}
    enc_maxlen=1000#训练时模型编码器、解码器总输入的最大长度（含cls、sep）
    profile_maxlen={
            'kdconv': 300,
            'hw': 300,
            }
    max_ut_num=16#最多保留多少句子
    dec_maxlen=512
    epoch=15
    lr_begin=2e-4#学习率
    lr_end=0
    batch_size=32
    data_size=164063
    step_per_epoch=data_size//batch_size#训练时是否带知识各占半
    warmup_step=step_per_epoch//2
    grad_accum_steps=1#梯度累积
    weight_decay=0.01#权重衰减率
    no_decay = ['bias', 'LayerNorm.weight']
    #eval相关
    global_step=0
    loss_momentum=0.999#loss滑动平均，便于显示，设0.99，前第220个值相对于当前值有0.1的影响力
    val_step=step_per_epoch//2#多少个batch验证一次
    bak_step=step_per_epoch#多少batch备份一次
    early_stop_step=val_step*5#多少次验证都没提高就早停
    last_better_step=0#上一次验证得分增加时的step
    best_val_score=0
    bestWeights=None#保存最好的模型的权重
    best_cp_loss=None#保存最好的模型checkpoint的loss
    start_val_epoch=0#多少epoch之后开始eval，可以节约eval时间
    start_val_loss=999#loss降到之后开始eval，可以节约eval时间
    #检索、生成解码
    max_dec_len=200#解码部分的最大长度，若希望太长截断可以设短，无限制可以设长（总长会自适应不必担心）
    min_dec_len=0
    do_sample=False#是否使用采样算法  随机采样
    sample_topk=2 #随机采样的个数
    num_beams=1#beam size
    num_return_sequences=1 #执行几次  得到几个句子
    p1_token='[user]'
    p2_token='[bot]'
    all_kg=None

