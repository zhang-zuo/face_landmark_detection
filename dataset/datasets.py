import os

import numpy as np
import cv2
import sys
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

sys.path.append('..')

from torch.utils import data
from torch.utils.data import DataLoader


class MyData(data.Dataset):

    def __init__(self, root_dir, image_dir, label_dir, transform=None):        # transform使用totensor和
        self.root_dir = root_dir                                          # 把路径给类的全局变量
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.label_path = os.path.join(self.root_dir, self.label_dir)     # 把根路径与图像或标签的文件夹 相连接
        self.image_path = os.path.join(self.root_dir, self.image_dir)
        self.image_list = os.listdir(self.image_path)                     # image_list 存放的就是image_path路径下的所有文件名
        self.label_list = os.listdir(self.label_path)
        self.transform = transform
        # 因为label 和 Image文件名相同，进行一样的排序，可以保证取出的数据和label是一一对应的
        self.image_list.sort()                                            # 对image_list 按照文件名排序
        self.label_list.sort()

    def __getitem__(self, idx):
        img_name = self.image_list[idx]                                   # 提取第idx个文件
        label_name = self.label_list[idx]
        img_item_path = os.path.join(self.root_dir, self.image_dir, img_name)   # 把第idx图像的路径放到img_item_path
        label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)
        img_item = Image.open(img_item_path)  # 打开图像

        label_txt = np.loadtxt(label_item_path,dtype=int,usecols=range(10), delimiter=',')  # 读取标签数组文件
        a = label_txt.reshape((5,2))
        label_resize = []
        for x,y in a:
            x_resize = x * (112 / img_item.size[0])
            y_resize = y * (112 / img_item.size[1])
            label_resize.append(int(x_resize))
            label_resize.append(int(y_resize))
        label = np.asarray(label_resize)

        trans_resize = transforms.Resize((112,112))
        img = trans_resize(img_item)

        if self.transform:
            img = self.transform(img)
        img = np.array(img)

        return img, label

    def __len__(self):
        return len(self.image_list)

class MyData_test(data.Dataset):

    def __init__(self, root_dir, image_dir, transform=None):        # transform使用totensor和
        self.root_dir = root_dir                                          # 把路径给类的全局变量
        self.image_dir = image_dir
        self.image_path = os.path.join(self.root_dir, self.image_dir)
        self.image_list = os.listdir(self.image_path)                     # image_list 存放的就是image_path路径下的所有文件名
        self.transform = transform
        # 因为label 和 Image文件名相同，进行一样的排序，可以保证取出的数据和label是一一对应的
        self.image_list.sort()                                            # 对image_list 按照文件名排序

    def __getitem__(self, idx):
        img_name = self.image_list[idx]                                   # 提取第idx个文件
        img_item_path = os.path.join(self.root_dir, self.image_dir, img_name)   # 把第idx图像的路径放到img_item_path
        img_item = Image.open(img_item_path)  # 打开图像

        trans_resize = transforms.Resize((112,112))
        img = trans_resize(img_item)

        if self.transform:
            img = self.transform(img)
        img = np.array(img)

        return img

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                    transforms.RandomHorizontalFlip(p=0.5),                     # 水平翻转  下面是光度
                                    transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=0.5)]), p=0.3)
                                    ])
    root_dir = "../data/train"
    image_train = "Images"
    label_train = "Annotations"
    ants_dataset = MyData(root_dir, image_train, label_train, transform)
    print(ants_dataset[0])

