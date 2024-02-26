from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.append("..")
sys.path.append(".")
import cv2
import os
import json
import pdb
import yaml
from pathlib import Path
from config_kpts import config_drone as cfg
import shutil
import data_util

def read_json(json_file):
    with open(json_file) as fp:
        json_data = json.load(fp)
        points = json_data['shapes'][0]
        points0 = json_data['shapes'][1]
        points1 = json_data['shapes'][2]
        points2 = json_data['shapes'][3]
        points3 = json_data['shapes'][4]
    pts = [points0['points'][0],points1['points'][0],points2['points'][0],points3['points'][0]]
    box = points['points']
    return pts,box

def read_phantom_json(json_file):
    pts = []
    with open(json_file) as fp:
        json_data = json.load(fp)
        for p in json_data['shapes']:
            pts.append(p['points'][0])
    return pts

def label_phantom():
    ##########################################
    file_name = '286'
    data_file_name = 'data_20230817_label'
    data_path = cfg['img_dir'] + data_file_name + '/' + file_name +'/'
    label_path = cfg['img_dir']+ data_file_name + '/' +  file_name + '/label/'
    save_path = cfg['img_dir'] + data_file_name + '/' +   file_name + '/check_label_res/'
    check_label = False
    ##########################################

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        shutil.rmtree(save_path,ignore_errors=True)
        os.makedirs(save_path)

    num_classes = cfg['num_classes']
    img_h = cfg['img_h']  ## 2160
    img_w = cfg['img_w']  ## 3840
    scale = cfg['scale']  ## 6
    color_vec = ['#2FEAF5', '#495B85', '#CFF975', '#C59447', '#28E4BC', '#44DE84', '#B4866D', '#DC143C', '#DED836']
    Path(data_path + "/unsupervised/").mkdir(parents=True, exist_ok=True)
    train_id_unsupervised_path = data_path + '/unsupervised/train_img_unsupervised_all.yml'
    train_id_path = data_path + '/supervised/train_img_supervised.yml'
    test_id_path = data_path + '/supervised/test_img.yml'
    
    files = os.listdir(data_path)
    img_files = []
    for f in files:
        if '.png' not in f:
            img_files.append(f)
    label_files = os.listdir(label_path)
    label_files.sort(key=lambda x: int(x.split(".json")[0]))
    train_set_num = len(files)
    print(train_set_num)
    
    training_img_id_list = data_util.generate_even_training_idx(train_set_num,3,start_from=1)
    train_unsupervised_id = []
    train_supervised_id = []
    test_id = []
    print(list(training_img_id_list))

    for i in label_files:
        i = int(i.split(".json")[0])
        pts = read_phantom_json(label_path + '{:0>6d}.json'.format(i))
        if np.shape(pts)[0] is not num_classes:
            continue
        pts_np = np.array(pts)
        pts_np = pts_np / scale
        hm = data_util.generate_hm(int(img_h / scale),int(img_w/ scale),pts_np)
        hm_total = hm[:,:,0]
        for k in range(0,num_classes):
            hm_total = hm_total + hm[:,:,k]
            Path(data_path + "/supervised/" + str(k)).mkdir(parents=True, exist_ok=True)
            np.save(data_path + "/supervised/" + str(k) + '/' + str(i) + '.npy',hm[:,:,k])
        if check_label:
            img = cv2.imread(data_path + "{:0>6d}.png".format(i))
            print(data_path + "{:0>6d}.png".format(i))
            for id, p in enumerate(pts):
                h = color_vec[id].lstrip('#')
                if int(p[0]) < 0 or int(p[1]) < 0:
                    continue
                cv2.circle(img,(int(p[0]),int(p[1])),6,tuple(int(h[i:i+2], 16) for i in (4, 2, 0)),-1)
                cv2.putText(img,str(id),(int(p[0]+2),int(p[1])),0,0.8,(255,0,0))
            plt.matshow(img)
            plt.show()
        print(f"label {i} is generated!")
        
    with open(train_id_path,'w') as file:
        yaml.dump(train_supervised_id,file)
    with open(train_id_unsupervised_path,'w') as file:
        yaml.dump(train_unsupervised_id,file)
    with open(test_id_path,'w') as file:
        yaml.dump(test_id,file)

def label_m300():
    file_name = '1681961122'
    data_file_name = 'data_20230420'
    data_path = cfg['img_dir'] + data_file_name + '/' + file_name +'/'
    label_path = cfg['img_dir'] + data_file_name + '/' +  file_name + '_label/' + file_name +'/'
    
    train_set_num = 1600
    training_img_id_list = data_util.generate_even_training_idx(train_set_num,10,start_from=1)
    
    ##########################################
    ######## check path and mkdir ############
    ##########################################
    Path(data_path + "/unsupervised/").mkdir(parents=True, exist_ok=True)
    train_id_unsupervised_path = data_path + '/unsupervised/train_img_unsupervised_all.yml'
    train_id_path = data_path + '/supervised/train_img_supervised.yml'
    test_id_path = data_path + '/supervised/test_img.yml'
    num_classes = 4 #cfg['num_classes']
    ##########################################
    img_files = os.listdir(data_path)
    label_files = os.listdir(label_path)
    color_img_files = []
    for i in img_files:
        if '.png' in i:
            color_img_files.append(i)

    color_img_files.sort(key=lambda x: float(x.split(".png")[0]))
    label_files.sort(key=lambda x: float(x.split(".json")[0]))
    label_time_stamp = []
    for i in label_files:
        i = float(i.split(".json")[0])
        label_time_stamp.append(i)
    train_unsupervised_id = []
    train_supervised_id = []
    test_id = []

    for i in color_img_files:
        i = float(i.split(".png")[0])
        print("image ",i)
        if i in label_time_stamp:
            id = label_time_stamp.index(i)
            pts,box = read_json(label_path + label_files[id] )
            pts_np = np.array(pts)
            pts_np = pts_np / 3
            hm = data_util.generate_hm(cfg["img_h"],cfg["img_w"],pts_np)
            hm_total = hm[:,:,0]
            for k in range(0,num_classes):
                hm_total = hm_total + hm[:,:,k]
                Path(data_path + "/supervised/" + str(k)).mkdir(parents=True, exist_ok=True)
                np.save(data_path + "/supervised/" + str(k) + '/' + str(i) + '.npy',hm[:,:,k])
            if id+1 not in training_img_id_list and id < train_set_num:
                train_unsupervised_id.append(i)
            elif id < train_set_num:
                train_supervised_id.append(i)
            elif id >= train_set_num:
                test_id.append(i)
    with open(train_id_path,'w') as file:
        yaml.dump(train_supervised_id,file)
    with open(train_id_unsupervised_path,'w') as file:
        yaml.dump(train_unsupervised_id,file)
    with open(test_id_path,'w') as file:
        yaml.dump(test_id,file)
    
    print("Done")

if __name__ == '__main__':
   ### This is for phantom

   label_phantom()
   
   ### This is for M300
   #label_m300()