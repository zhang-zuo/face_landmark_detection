#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import logging
import warnings
import random
import os
import torch
import numpy as np


from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from config import get_config
from dataset.datasets import MyData
from pfld.utils import init_weights, save_checkpoint, set_logger, write_cfg
from pfld.loss import LandmarkLoss
# from test import compute_nme

from models.PFLD import PFLD
from models.PFLD_Ultralight import PFLD_Ultralight
from models.PFLD_Ultralight_Slim import PFLD_Ultralight_Slim


def train(model, train_dataloader, loss_fn, optimizer, cfg):
    losses = []
    model.train()
    with tqdm(total=len(train_dataloader)) as t:           # tqdm进度条显示
        for img, landmark_gt in train_dataloader:
            img = img.to(cfg.DEVICE)
            landmark_gt = landmark_gt.to(cfg.DEVICE)
            landmark_pred = model(img)
            loss = loss_fn(landmark_gt, landmark_pred)
            optimizer.zero_grad()                          # 清空梯度
            loss.backward()                                # 计算梯度
            optimizer.step()                               # 根据梯度更新权重

            losses.append(loss.cpu().detach().numpy())     # 把损失放进losses列表中
            t.update()

    return np.mean(losses)                                 # 把所有train的数据进行训练后的损失均值返回


def validate(model, val_dataloader, loss_fn, cfg):
    model.eval()
    losses = []
    rmse_list = []
    with torch.no_grad():                                          # 表明这些不需要跟踪梯度
        for img, landmark_gt in val_dataloader:
            img = img.to(cfg.DEVICE)
            landmark_gt = landmark_gt.to(cfg.DEVICE)
            landmark_pred = model(img)
            loss = loss_fn(landmark_gt, landmark_pred)
            losses.append(loss.cpu().numpy())

            landmark_pred = landmark_pred.reshape((-1,10)).cpu().numpy()
            landmark_gt = landmark_gt.reshape((-1,10)).cpu().numpy()

            rmse_item = []
            for num in range(landmark_pred.shape[0]):
                a = landmark_pred[num,:]
                b = landmark_gt[num,:]
                rmse_temp = np.sqrt(((a - b) ** 2).mean())
                rmse_item.append(rmse_temp)

            rmse_it = np.mean(rmse_item)
            rmse_list.append(rmse_it)

    return np.mean(losses), np.mean(rmse_list)



def main():
    cfg = get_config()

    SEED = cfg.SEED                                 # seed随机数种子
    np.random.seed(SEED)                            # 这块是将模型生成的随机数每次都相同，就会得到相同的结果
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    warnings.filterwarnings("ignore")
    set_logger(cfg.LOGGER_PATH)
    write_cfg(logging, cfg)

    main_worker(cfg)


def main_worker(cfg):
    # ======= LOADING DATA ======= #
    logging.warning('=======>>>>>>> Loading Training and Validation Data')
    ROOT_TRAIN_PATH = cfg.ROOT_TRAIN_PATH
    ROOT_TEST_PATH = cfg.ROOT_TEST_PATH
    IMAGE_PATH = cfg.IMAGE_PATH
    LABEL_PATH = cfg.LABEL_PATH
    TRANSFORM = cfg.TRANSFORM

    train_dataset = MyData(ROOT_TRAIN_PATH,IMAGE_PATH,LABEL_PATH, TRANSFORM)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAIN_BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=False)

    val_dataset = MyData(ROOT_TEST_PATH,IMAGE_PATH,LABEL_PATH, TRANSFORM)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.VAL_BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

    # ======= MODEL ======= #
    MODEL_DICT = {'PFLD': PFLD,
                  'PFLD_Ultralight': PFLD_Ultralight,
                  'PFLD_Ultralight_Slim': PFLD_Ultralight_Slim,
                  }
    MODEL_TYPE = cfg.MODEL_TYPE                      # PFLD_Ultralight
    WIDTH_FACTOR = cfg.WIDTH_FACTOR                  # 0.25
    INPUT_SIZE = cfg.INPUT_SIZE                      # 112*112
    LANDMARK_NUMBER = cfg.LANDMARK_NUMBER            # 5
    model = MODEL_DICT[MODEL_TYPE](WIDTH_FACTOR, INPUT_SIZE[0], LANDMARK_NUMBER).to(cfg.DEVICE)
    # cfg.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   可选的设备cpu或者cuda

    if cfg.RESUME:
        if os.path.isfile(cfg.RESUME_MODEL_PATH):
            model.load_state_dict(torch.load(cfg.RESUME_MODEL_PATH))
        else:
            logging.warning("MODEL: No Checkpoint Found at '{}".format(cfg.RESUME_MODEL_PATH))
    logging.warning('=======>>>>>>> {} Model Generated'.format(MODEL_TYPE))

    # 使用wing loss
    # ======= LOSS ======= #
    loss_fn = LandmarkLoss(LANDMARK_NUMBER)
    logging.warning('=======>>>>>>> Loss Function Generated')

    # 使用adam优化器，对学习率进行动态调整
    # ======= OPTIMIZER ======= #
    optimizer = torch.optim.Adam(
        [{'params': model.parameters()}],
        lr=cfg.LR,                                       # 默认的学习率
        weight_decay=cfg.WEIGHT_DECAY)                   # 权重衰减
    logging.warning('=======>>>>>>> Optimizer Generated')

    # cfg.MILESTONES = [90, 140, 170]
    # 在epoch达到90，140，170的时候，让学习率乘以gamma进行
    # ======= SCHEDULER ======= #
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.MILESTONES, gamma=0.1)
    logging.warning('=======>>>>>>> Scheduler Generated' + '\n')

    # ======= TENSORBOARDX WRITER ======= #
    writer = SummaryWriter(cfg.LOG_PATH)

    # 生成一个假的图片，给add_graph，用于生成模型图表
    #dummy_input = torch.rand(1, 3, INPUT_SIZE[0], INPUT_SIZE[1]).to(cfg.DEVICE)
    #writer.add_graph(model, (dummy_input,))

    #best_nme = float('inf')
    for epoch in range(1, cfg.EPOCHES + 1):
        logging.warning('Epoch {} Start'.format(epoch))
        train_loss = train(model, train_dataloader, loss_fn, optimizer, cfg)
        val_loss, val_rmse = validate(model, val_dataloader, loss_fn, cfg)
        scheduler.step()

        #if val_rmse < 1:                   # 使用RMSE的值进行筛选，当做阈值
        #    save_checkpoint(cfg, model, extra='best')
        #    logging.info('Save best model')
        #save_checkpoint(cfg, model, epoch)

        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Train_Loss', train_loss, epoch)
        writer.add_scalar('Val_Loss', val_loss, epoch)
        #writer.add_scalar('Val_RMSE -> seed={} batch_size={} epoch={} lr={} weight_decay={} MILESTONES={}'.format(cfg.SEED,cfg.TRAIN_BATCH_SIZE,cfg.EPOCHES,cfg.LR,cfg.WEIGHT_DECAY,cfg.MILESTONES), val_rmse, epoch)
        writer.add_scalar('Val_RMSE', val_rmse, epoch)

        logging.info('Train_Loss: {}'.format(train_loss))
        logging.info('Val_Loss: {}'.format(val_loss))
        logging.info('Val_RMSE: {}'.format(val_rmse) + '\n')

    save_checkpoint(cfg, model, extra='final')


if __name__ == "__main__":
    main()
