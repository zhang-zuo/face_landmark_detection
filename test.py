import argparse
import os
import time

import cv2
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataset.datasets import MyData, MyData_test

from config import get_config

from models.PFLD import PFLD
from models.PFLD_Ultralight import PFLD_Ultralight
from models.PFLD_Ultralight_Slim import PFLD_Ultralight_Slim
from torch.utils.tensorboard import SummaryWriter

#cudnn.benchmark = True
#cudnn.determinstic = True
#cudnn.enabled = True

def validate_test(model, pre_dataloader, args):
    model.eval()
    writer = SummaryWriter("./checkpoint/predict_log/")

    cost_time = []

    img_list = os.listdir(args.predict_path)
    img_list.sort()

    with torch.no_grad():
        idx = 0
        for img_112 in pre_dataloader:
            img_112 = img_112.to(args.device)

            img_name = img_list[idx]
            img_path = os.path.join(args.predict_path,img_name)

            start_time = time.time()
            landmarks_pre_112 = model(img_112)
            #print(next(model.parameters()).device)


            landmarks_pre_112 = landmarks_pre_112.cpu().numpy()

            img_basic = cv2.imread(img_path)

            landmarks_pre_112_count = np.reshape(landmarks_pre_112,(-1,2))
            idx_lm = 0
            landmark_pre_basic_count = np.empty((5,2))
            for x,y in landmarks_pre_112_count:
                x = x*(img_basic.shape[0]/args.input_size)
                y = y*(img_basic.shape[1]/args.input_size)
                landmark_pre_basic_count[idx_lm,:] = x,y
                idx_lm += 1


            if args.show_image_112:
                show_img = np.array(np.transpose((img_112[0] * 0.5 + 0.5).cpu().numpy(), (1, 2, 0)))
                show_img = (show_img * 255).astype(np.uint8)
                show_img_112 = show_img.copy()

                landmarks_pre_112_show = np.reshape(landmarks_pre_112,(-1,2))
                for x,y in landmarks_pre_112_show:
                    cv2.circle(show_img_112, (int(x), int(y)), 1, (0, 0, 255), -1)
                cv2.imwrite("./result/Images_112/{}.jpg".format(idx), show_img_112)                       # 保存图像

            if args.show_label_basic:
                name, suffix = os.path.splitext(img_name)
                landmark_save_txt = np.reshape(landmark_pre_basic_count,(1,-1))
                np.savetxt("./result/Landmark_basic/{}.txt".format(name), landmark_save_txt, fmt='%1.5g',delimiter=',')

            if args.show_image_basic:
                landmarks_pre_basic = np.reshape(landmark_pre_basic_count,(-1,2))
                for x,y in landmarks_pre_basic:
                    cv2.circle(img_basic, (int(x), int(y)), 1, (0, 0, 255), -1)
                cv2.imwrite("./result/Images/{}.jpg".format(idx), img_basic)

            cost_time.append(time.time() - start_time)
            writer.add_scalar("times:",(time.time() - start_time),idx)

            idx += 1
    return np.mean(cost_time)


def validate(model, pre_dataloader, args):
    model.eval()
    writer = SummaryWriter("./checkpoint/predict_log/")

    rmse_pre_112 = []
    rmse_pre_basic = []
    cost_time = []

    img_list = os.listdir(args.predict_path)
    label_list = os.listdir(args.predict_label_path)
    img_list.sort()
    label_list.sort()

    with torch.no_grad():
        idx = 0
        for img_112, landmark_gt in pre_dataloader:
            img_112 = img_112.to(args.device)
            #landmark_gt_112 = landmark_gt_112.to(args.device)

            img_name = img_list[idx]
            label_name = label_list[idx]
            img_path = os.path.join(args.predict_path,img_name)
            label_path = os.path.join(args.predict_label_path,label_name)

            start_time = time.time()
            landmarks_pre_112 = model(img_112)
            #print(next(model.parameters()).device)


            landmarks_pre_112 = landmarks_pre_112.cpu().numpy()
            #landmark_gt_112 = landmark_gt_112.cpu().numpy()

            img_basic = cv2.imread(img_path)
            landmark_gt_basic_txt = np.loadtxt(label_path,dtype=int,usecols=range(10), delimiter=',')

            landmarks_pre_112_count = np.reshape(landmarks_pre_112,(-1,2))
            idx_lm = 0
            landmark_pre_basic_count = np.empty((5,2))
            for x,y in landmarks_pre_112_count:
                x = x*(img_basic.shape[0]/args.input_size)
                y = y*(img_basic.shape[1]/args.input_size)
                #print(x,y)
                landmark_pre_basic_count[idx_lm,:] = x,y
                idx_lm += 1

            landmark_gt_basic_count = np.reshape(landmark_gt_basic_txt,(-1,2))
            idx_lm = 0
            landmark_gt_112_count = np.empty((5,2))
            for x,y in landmark_gt_basic_count:
                x = x*(args.input_size/img_basic.shape[0])
                y = y*(args.input_size/img_basic.shape[1])
                #print(x,y)
                landmark_gt_112_count[idx_lm,:] = x,y
                idx_lm += 1

            if args.rmse_count_112:
                landmarks_pre_112 = np.reshape(landmarks_pre_112,(1,-1))
                landmark_gt_112 = np.reshape(landmark_gt_112_count,(1,-1))
                rmse_pre = np.sqrt(((landmarks_pre_112 - landmark_gt_112) ** 2).mean())
                rmse_pre_112.append(rmse_pre)
                #print(rmse_pre)
                writer.add_scalar("RMSE_112:",rmse_pre,idx)

            if args.show_image_112:
                show_img = np.array(np.transpose((img_112[0] * 0.5 + 0.5).cpu().numpy(), (1, 2, 0)))
                show_img = (show_img * 255).astype(np.uint8)
                show_img_112 = show_img.copy()

                landmarks_pre_112_show = np.reshape(landmarks_pre_112,(-1,2))
                landmark_gt_112_show = np.reshape(landmark_gt_112_count,(-1,2))
                for x,y in landmarks_pre_112_show:
                    cv2.circle(show_img_112, (int(x), int(y)), 1, (0, 0, 255), -1)
                for x,y in landmark_gt_112_show:
                    cv2.circle(show_img_112, (int(x), int(y)), 1, (0, 255, 0), -1)
                cv2.imwrite("./result/Images_112/{}.jpg".format(idx), show_img_112)                       # 保存图像
                #print("save:{}".format(idx))


            if args.show_label_basic:
                name, suffix = os.path.splitext(img_name)
                landmark_save_txt = np.reshape(landmark_pre_basic_count,(1,-1))
                np.savetxt("./result/Landmark_basic/{}.txt".format(name), landmark_save_txt, fmt='%1.5g',delimiter=',')

            if args.rmse_count_basic:
                landmark_pre_basic = np.reshape(landmark_pre_basic_count,(1,-1))
                landmark_gt_basic = np.reshape(landmark_gt_basic_txt,(1,-1))
                rmse_basic = np.sqrt(((landmark_pre_basic - landmark_gt_basic) ** 2).mean())
                rmse_pre_basic.append(rmse_basic)
                writer.add_scalar("RMSE_basic:",rmse_basic,idx)

            if args.show_image_basic:
                landmarks_pre_basic = np.reshape(landmark_pre_basic_count,(-1,2))
                landmark_gt_basic = np.reshape(landmark_gt_basic_txt,(-1,2))

                for x,y in landmarks_pre_basic:
                    cv2.circle(img_basic, (int(x), int(y)), 1, (0, 0, 255), -1)
                for x,y in landmark_gt_basic:
                    cv2.circle(img_basic, (int(x), int(y)), 1, (0, 255, 0), -1)

                cv2.imwrite("./result/Images/{}.jpg".format(idx), img_basic)

            cost_time.append(time.time() - start_time)
            writer.add_scalar("times:",(time.time() - start_time),idx)

            idx += 1
    return np.mean(cost_time), np.mean(rmse_pre_112), np.mean(rmse_pre_basic)


def main(args):
    MODEL_DICT = {'PFLD': PFLD,
                  'PFLD_Ultralight': PFLD_Ultralight,
                  'PFLD_Ultralight_Slim': PFLD_Ultralight_Slim,
                  }
    MODEL_TYPE = args.model_type
    WIDTH_FACTOR = args.width_factor
    INPUT_SIZE = args.input_size
    LANDMARK_NUMBER = args.landmark_number
    model = MODEL_DICT[MODEL_TYPE](WIDTH_FACTOR, INPUT_SIZE, LANDMARK_NUMBER).to(args.device)

    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    ROOT_TEST_PATH = './data/predict'
    IMAGE_PATH = 'Images'
    LABEL_PATH = 'Annotations'
    #wlfw_val_dataset = WLFWDatasets(args.test_dataset, transform)


    if is_exist_file(ROOT_TEST_PATH, LABEL_PATH):
        #print(is_exist_file(label_path))
        pre_dataset = MyData(ROOT_TEST_PATH, IMAGE_PATH, LABEL_PATH, transform)
        pre_dataloader = DataLoader(pre_dataset, batch_size=1, shuffle=False, num_workers=8)
        mean_time, rmse_112, rmse_basic = validate(model, pre_dataloader, args)
    else:
        #print(is_exist_file(label_path))
        pre_dataset = MyData_test(ROOT_TEST_PATH, IMAGE_PATH, transform)
        pre_dataloader = DataLoader(pre_dataset, batch_size=1, shuffle=False, num_workers=8)
        mean_time = validate_test(model, pre_dataloader, args)

    print("一共{}张图像参与预测：".format(len(pre_dataset)))
    print("平均每张图片用时：{}ms".format(mean_time*1000))
    if is_exist_file(ROOT_TEST_PATH, LABEL_PATH):
        print("平均的RMSE_112值：{}".format(rmse_112))
        print("平均的RMSE_basic值：{}".format(rmse_basic))


    #data = torch.ones([2, 3])
    #print(data.device)  # 输出：cpu


def parse_args():
    cfg = get_config()
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_type', default='PFLD_Ultralight', type=str)
    parser.add_argument('--input_size', default=112, type=int)
    parser.add_argument('--width_factor', default=0.25, type=float)
    parser.add_argument('--landmark_number', default=5, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--model_path', default="./checkpoint/weight/pfld_ultralight_final.pth", type=str)
    #parser.add_argument('--test_dataset', default='./data/test_data/list.txt', type=str)
    parser.add_argument('--predict_path', default="./data/predict/Images/", type=str)
    parser.add_argument('--predict_label_path', default="./data/predict/Annotations/", type=str)

    parser.add_argument('--show_image_112', default=False, type=bool)
    parser.add_argument('--rmse_count_112', default=True, type=bool)
    parser.add_argument('--show_label_basic', default=True, type=bool)
    parser.add_argument('--show_image_basic', default=True, type=bool)
    parser.add_argument('--rmse_count_basic', default=True, type=bool)

    args = parser.parse_args()
    return args

def is_exist_file(root_path, file_path):
    label_path = os.path.join(root_path,file_path)
    is_exixt = os.path.exists(label_path)
    if is_exixt == False:
        return False
    Files=os.listdir(label_path)
    for k in range(len(Files)):
        Files[k]=os.path.splitext(Files[k])[1]
    # 你想要找的文件的后缀
    Str='.txt'
    if Str in Files:
        return True
    else:
        return False



if __name__ == "__main__":
    args = parse_args()
    main(args)
