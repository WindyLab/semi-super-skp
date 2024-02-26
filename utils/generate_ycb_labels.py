from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import yaml
import sys
sys.path.append("..")
sys.path.append(".")
import pdb
from config_kpts import config_ycb as cfg
import data_util
from pathlib import Path
import scipy.io as scio

def generate_heatmap_label():
    data_set_id = cfg['data_set_ids']
    obj_index = cfg['obj_index']
    sigma = 2
    num_classes = cfg['num_classes']
    
    for data_set in data_set_id:
        data_path = cfg['img_dir']+ "{:0>4d}".format(data_set) + '/'
        train_set_num = 1500
        
        ## Generate training index for supervised data. Use invertal to control ratio.
        training_img_id_list = data_util.generate_even_training_idx(train_set_num,10,start_from=1)
        
        ##########################################
        ######## check path and mkdir ############
        ##########################################
        Path(data_path + "/unsupervised/").mkdir(parents=True, exist_ok=True)
        train_id_unsupervised_path = data_path + '/unsupervised/train_img_unsupervised_all.yml'
        train_id_path = data_path + '/supervised/train_img_supervised.yml'
        test_id_path = data_path + '/supervised/test_img.yml'
        ##########################################

        files = os.listdir(data_path)
        color_img_files = []
        depth_img_files = []
        meta_files = []
        for f in files:
            if '-color' in f:
                color_img_files.append(f)
            if '-depth' in f:
                depth_img_files.append(f)
            if '-meta' in f:
                meta_files.append(f)
        color_img_files.sort(key=lambda x: float(x.split("-color")[0]))
        depth_img_files.sort(key=lambda x: float(x.split("-depth")[0]))
        meta_files.sort(key=lambda x: float(x.split("-meta")[0]))
        print("color_img_files:",len(color_img_files))
        print("depth_img_files:",len(depth_img_files))
        print("meta_files:",len(meta_files))
        
        assert len(color_img_files) == len(depth_img_files) == len(meta_files)
    
        num_data  = len(color_img_files)
        print("num_data:",num_data)
        pts = np.array(
            cfg['pts3d']
        )

        train_unsupervised_id = []
        train_supervised_id = []
        test_id = []
        observations = []
        for id,meta_file in enumerate(meta_files):
            print("data id ",id)
            data = scio.loadmat(data_path + meta_file)
            cls_indexes = data['cls_indexes']

            obj_i = np.where(cls_indexes == obj_index)[0][0]

            factor_depth = data['factor_depth']
            r_cam = data['poses'][:,:,obj_i][0:3,0:3]
            t_cam = data['poses'][:,:,obj_i][:,3]
            
            # x_in_cam = r_cam @ x + t_cam
            # y_in_cam = r_cam @ y + t_cam
            # z_in_cam = r_cam @ z + t_cam
            
            # ux = data['intrinsic_matrix'][0][0] * x_in_cam[0]/x_in_cam[2] + data['intrinsic_matrix'][0][2]
            # vx = data['intrinsic_matrix'][1][1] * x_in_cam[1]/x_in_cam[2] + data['intrinsic_matrix'][1][2]
            
            # uy = data['intrinsic_matrix'][0][0] * y_in_cam[0]/y_in_cam[2] + data['intrinsic_matrix'][0][2]
            # vy = data['intrinsic_matrix'][1][1] * y_in_cam[1]/y_in_cam[2] + data['intrinsic_matrix'][1][2]
            
            # uz = data['intrinsic_matrix'][0][0] * z_in_cam[0]/z_in_cam[2] + data['intrinsic_matrix'][0][2]
            # vz = data['intrinsic_matrix'][1][1] * z_in_cam[1]/z_in_cam[2] + data['intrinsic_matrix'][1][2]
            
            # u0 = data['intrinsic_matrix'][0][0] * t_cam[0]/t_cam[2] + data['intrinsic_matrix'][0][2]
            # v0 = data['intrinsic_matrix'][0][0] * t_cam[1]/t_cam[2] + data['intrinsic_matrix'][1][2]

            img = cv2.imread(data_path + color_img_files[id])
            depth = cv2.imread(data_path + depth_img_files[id],cv2.CV_16UC1)
            
            img_rz = cv2.resize(img,(cfg['img_w'],cfg['img_h']))
            depth_rz = cv2.resize(depth,(cfg['img_w'],cfg['img_h']))
            img_h = img_rz.shape[0]
            img_w = img_rz.shape[1]
            #print(img_h,img_w)

            pro_pts = np.zeros((num_classes,2))
            pro_pts += -1
            for pts_id,i in enumerate(pts):
                p_in_cam = r_cam @ i + t_cam
                u = data['intrinsic_matrix'][0][0] / 2.0 * p_in_cam[0]/p_in_cam[2] + data['intrinsic_matrix'][0][2] / 2.0 
                v = data['intrinsic_matrix'][1][1] / 2.0 * p_in_cam[1]/p_in_cam[2] + data['intrinsic_matrix'][1][2] / 2.0
                if u < 0 or u >= img_rz.shape[1] or v < 0 or v >= img_rz.shape[0]:
                    continue
                depth_value = depth_rz[int(v)][int(u)] / factor_depth
                if pts_id > 3 and p_in_cam[2] - depth_value[0][0] > 0.03 and abs(depth_value[0][0] - 0.0) > 1e-05:
                    pro_pts[pts_id][0] = -1
                    pro_pts[pts_id][1] = -1
                    continue
                cv2.circle(img_rz,(int(u),int(v)),1,(255,0,0),-1)
                cv2.putText(img_rz,str(pts_id),(int(u),int(v)),0,1,(0,200,0))
                pro_pts[pts_id][0] = u
                pro_pts[pts_id][1] = v

            hm = data_util.generate_hm(img_h,img_w,pro_pts,sigma)
            hm_total = hm[:,:,0]
            for k in range(0,num_classes):
                hm_total = hm_total + hm[:,:,k]
                Path(data_path + "/supervised/" + str(k)).mkdir(parents=True, exist_ok=True)
                np.save(data_path + "/supervised/" + str(k) + '/' + "{:0>6d}".format(id) + '.npy',hm[:,:,k])
            if id+1 not in training_img_id_list and id < train_set_num:
                train_unsupervised_id.append(id+1)
            elif id < train_set_num:
                train_supervised_id.append(id+1)
            elif id >= train_set_num:
                test_id.append(id+1)
        with open(train_id_path,'w') as file:
            yaml.dump(train_supervised_id,file)
        with open(train_id_unsupervised_path,'w') as file:
            yaml.dump(train_unsupervised_id,file)
        with open(test_id_path,'w') as file:
            yaml.dump(test_id,file)

def get_uv_from_Ps(p,Ps):
    pro = Ps @ np.concatenate((p,[1]),axis=0)
    u = pro[0] / pro[2]
    v = pro[1] / pro[2]
    return u,v 

def get_ycb_model_id(obj_names):
    model_names = os.listdir('/media/datadisk/data_space/YCB_Video_Dataset/models')
    for id,name in enumerate(model_names):
        obj_names[id+1] = name

def get_scene_obj():
    obj_names = {}
    model_names = os.listdir('/media/datadisk/data_space/YCB_Video_Dataset/models')
    for id,name in enumerate(model_names):
        obj_names[id+1] = name
    for scene_id in range(0,91):
        scene_path = '/media/datadisk/data_space/YCB_Video_Dataset/data/{:04d}/'.format(scene_id)
        if not os.path.exists(scene_path):
            continue
        files = os.listdir(scene_path)
        meta_files = []
        for f in files:
            if '-meta' in f:
                meta_files.append(f)
        meta_files.sort(key=lambda x: float(x.split("-meta")[0]))
        data = scio.loadmat(scene_path + meta_files[0])
        cls_indexes = data['cls_indexes']
        print('scene id:{:04d}'.format(scene_id))
        for id in cls_indexes:
           print(id,obj_names[id[0]])
        print('\n')

if __name__ == '__main__':
    generate_heatmap_label()