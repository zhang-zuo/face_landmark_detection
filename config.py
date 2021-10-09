from torchvision import transforms as trans
from easydict import EasyDict as edict
from pfld.utils import get_time
import os
import torch


def get_config():
    cfg = edict()
    cfg.SEED = 10
    cfg.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg.TRANSFORM = trans.Compose([trans.ToTensor(),
                                   trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                   #trans.RandomHorizontalFlip(p=0.1)                   # 水平翻转  下面是光度
                                   #trans.RandomApply(torch.nn.ModuleList([trans.ColorJitter(brightness=0.5)]), p=0.05)
                                   ])

    cfg.MODEL_TYPE = 'PFLD_Ultralight'  # [PFLD, PFLD_Ultralight, PFLD_Ultralight_Slim]
    cfg.INPUT_SIZE = [112, 112]
    cfg.WIDTH_FACTOR = 0.25
    cfg.LANDMARK_NUMBER = 5

    cfg.TRAIN_BATCH_SIZE = 32
    cfg.VAL_BATCH_SIZE = 8

    cfg.ROOT_TRAIN_PATH = './data/train'
    cfg.ROOT_TEST_PATH = './data/validation'
    cfg.IMAGE_PATH = 'Images'
    cfg.LABEL_PATH = 'Annotations'

    cfg.EPOCHES = 100
    cfg.LR = 1e-2
    cfg.WEIGHT_DECAY = 1e-3
    cfg.NUM_WORKERS = 8
    cfg.MILESTONES = [20, 50, 75]

    cfg.RESUME = False
    if cfg.RESUME:
        cfg.RESUME_MODEL_PATH = ''

    create_time = get_time()
    #cfg.MODEL_PATH = './checkpoint/weight/{}_{}_{}_{}/'.format(cfg.MODEL_TYPE, cfg.WIDTH_FACTOR, cfg.INPUT_SIZE[0], create_time)
    #cfg.MODEL_PATH = './checkpoint/weight/{}_{}_{}/'.format(cfg.MODEL_TYPE, cfg.WIDTH_FACTOR, cfg.INPUT_SIZE[0])
    cfg.MODEL_PATH = './checkpoint/weight/'

    cfg.LOG_PATH = './checkpoint/log/'
    #cfg.LOG_PATH = './checkpoint/log/{}_{}_{}_{}/'.format(cfg.MODEL_TYPE, cfg.WIDTH_FACTOR, cfg.INPUT_SIZE[0], create_time)
    cfg.LOGGER_PATH = os.path.join(cfg.LOG_PATH, "train.log")
    if not os.path.exists(cfg.MODEL_PATH):
        os.makedirs(cfg.MODEL_PATH)

    return cfg
