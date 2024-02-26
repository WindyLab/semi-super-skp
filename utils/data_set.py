import numpy as np
import cv2
import os
import yaml
from torch.utils.data import Dataset
import torch
import sys
sys.path.append("..")
import logging
from torchvision import transforms
from config_kpts import config_ycb as cfg

class YCBDataset(Dataset):
    def __init__(self, images_dir: str, hms_dir: str, train_info_dir:str, num_classes,scale: float = 1.0):
        self.images_dir = images_dir
        self.hms_dir = hms_dir
        hms_directories = os.listdir(hms_dir)
        
        self.train_info_dir = train_info_dir
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.num_classes = num_classes
        with open(self.train_info_dir, 'r') as f:
            train_info = yaml.load(f, Loader=yaml.CLoader)
      #  print("hms_dir:",hms_dir)
     #   print("train_info:",train_info)
        
        self.img_files = self.filter_imgs(images_dir,train_info)
        self.hms_files = self.filter_hms(hms_dir + '/' + hms_directories[0] + '/', train_info)
      #  print("img_files:",self.img_files)
     #   print("hms_files:",self.hms_files)
        
        self.train_info = train_info
        self.data_num = len(train_info)
        num_hms = len(self.hms_files)
        num_img = len(self.img_files)
        assert num_hms == num_img
        logging.info(f'Creating dataset with {num_hms} / {num_img} examples')

    def filter_imgs(self,img_dir,info):
        files = os.listdir(img_dir)
        color_img_files = []
        for f in files:
            if '-color' in f and int(f.split("-color")[0]) in info:
                color_img_files.append(f)
        color_img_files.sort(key=lambda x: float(x.split("-color")[0]))
        return color_img_files
    
    def filter_hms(self,hms_dir,info):
        files = os.listdir(hms_dir)
        hms_files = []
        for f in files:
            if int(f.split(".")[0]) in info:
                hms_files.append(f)
        hms_files.sort(key=lambda x: float(x.split(".")[0]))
        return hms_files

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        #read original image
        img_name = self.images_dir + self.img_files[idx]
        #print(img_name)
        orig_img = cv2.imread(img_name)
        img = cv2.resize(orig_img,(cfg['img_w'],cfg['img_h']))
        if np.random.randint(0,2) == 0:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if np.random.randint(0,10) == 0:
            img = self.preprocess(img)

        if np.random.randint(0,2) == 1:
            img = self.add_noise(img)
        img = transforms.ToTensor()(img)
        img = img.to(torch.float32)
        
        ######  read heatmap ############
        heat_map_list = []
        for cl in range(0,self.num_classes):
            hms_name = self.hms_dir + '/' + str(cl) + '/' + self.hms_files[idx]
            heatmap = np.load(hms_name,allow_pickle=True)
            heatmap = torch.tensor(heatmap, dtype=torch.float32)
            heatmap = heatmap.unsqueeze(dim=0)
            heat_map_list.append(heatmap)
        heat_map_list = torch.stack(heat_map_list,dim=0)
        heat_map_list = heat_map_list.squeeze()
        return img,heat_map_list

    def preprocess(self, data):
        # random hue and saturation
        img_hsv = cv2.cvtColor(data, cv2.COLOR_RGB2HSV)
        img_hsv = img_hsv.astype(np.float64)
        delta = (np.random.random() * 2 - 1) * 0.2
        img_hsv[:, :, 0] = np.mod(data[:,:,0] + (delta * 360 + 360.), 360.)

        delta_sature = np.random.random() + 0.5
        img_hsv[:, :, 1] *= delta_sature
        img_hsv[:,:, 1] = np.maximum( np.minimum(data[:,:,1], 1), 0 )
        img_hsv = img_hsv * 255.0
        img_hsv = img_hsv.astype(np.uint8)
        img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        # adjust brightness
        img_rgb = img_rgb.astype(np.float64)
        delta = (np.random.random() * 2 - 1) * 0.3
        img_rgb += delta
        return img_rgb.astype(np.uint8)
    
    def add_noise(self,img):
        row,col,ch = img.shape
        gauss = np.random.normal(0,0.9,(row,col,ch))
        img = img.astype(np.float64) + gauss
        return img.astype(np.uint8)


class DroneDataset(Dataset):
    def __init__(self, images_dir: str, hms_dir: str, train_info_dir:str, num_classes,conf,scale: float = 1.0):
        self.images_dir = images_dir
        self.hms_dir = hms_dir
        hms_directories = os.listdir(hms_dir)
        self.conf = conf
        self.train_info_dir = train_info_dir
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.num_classes = num_classes
        with open(self.train_info_dir, 'r') as f:
            train_info = yaml.load(f, Loader=yaml.CLoader)
        print("hms_dir:",hms_dir)
        print("train_info:",train_info)
        
        self.img_files = self.filter_imgs(images_dir,train_info)
        self.hms_files = self.filter_hms(hms_dir + '/' + hms_directories[0] + '/', train_info)
        print("img_files:",self.img_files)
        print("hms_files:",self.hms_files)

        self.train_info = train_info
        self.data_num = len(train_info)
        num_hms = len(self.hms_files)
        num_img = len(self.img_files)
        assert num_hms == num_img
        logging.info(f'Creating dataset with {num_hms} / {num_img} examples')

    def filter_imgs(self,img_dir,info):
        files = os.listdir(img_dir)
        color_img_files = []
        for f in files:
            if '.png' in f and float(f.split(".png")[0]) in info:
                color_img_files.append(f)
        color_img_files.sort(key=lambda x: float(x.split(".png")[0]))
        return color_img_files
    
    def filter_hms(self,hms_dir,info):
        files = os.listdir(hms_dir)
        hms_files = []
        for f in files:
            if int(f.split(".npy")[0]) in info:
                hms_files.append(f)
        hms_files.sort(key=lambda x: int(x.split(".")[0]))
        return hms_files

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        #read original image
        img_name = self.images_dir + self.img_files[idx]
       # print(img_name)
        orig_img = cv2.imread(img_name)
        img = cv2.resize(orig_img,(self.conf.train.img_w,self.conf.train.img_h))
        if np.random.randint(0,2) == 0:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if np.random.randint(0,10) == 0:
            img = self.preprocess(img)

        if np.random.randint(0,2) == 1:
            img = self.add_noise(img)
        img = transforms.ToTensor()(img)
        img = img.to(torch.float32)
        
        ######  read heatmap ############
        heat_map_list = []
        for cl in range(0,self.num_classes):
            hms_name = self.hms_dir + '/' + str(cl) + '/' + self.hms_files[idx]
            heatmap = np.load(hms_name,allow_pickle=True)
            heatmap = torch.tensor(heatmap, dtype=torch.float32)
            heatmap = heatmap.unsqueeze(dim=0)
            heat_map_list.append(heatmap)
        heat_map_list = torch.stack(heat_map_list,dim=0)
        heat_map_list = heat_map_list.squeeze()
        return img,heat_map_list

    def preprocess(self, data):
        # random hue and saturation
        img_hsv = cv2.cvtColor(data, cv2.COLOR_RGB2HSV)
        img_hsv = img_hsv.astype(np.float64)
        delta = (np.random.random() * 2 - 1) * 0.2
        img_hsv[:, :, 0] = np.mod(data[:,:,0] + (delta * 360 + 360.), 360.)

        delta_sature = np.random.random() + 0.5
        img_hsv[:, :, 1] *= delta_sature
        img_hsv[:,:, 1] = np.maximum( np.minimum(data[:,:,1], 1), 0 )
        img_hsv = img_hsv * 255.0
        img_hsv = img_hsv.astype(np.uint8)
        img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        # adjust brightness
        img_rgb = img_rgb.astype(np.float64)
        delta = (np.random.random() * 2 - 1) * 0.3
        img_rgb += delta
        return img_rgb.astype(np.uint8)
    
    def add_noise(self,img):
        row,col,ch = img.shape
        gauss = np.random.normal(0,0.9,(row,col,ch))
        img = img.astype(np.float64) + gauss
        return img.astype(np.uint8)