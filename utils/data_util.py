import numpy as np
import yaml
import scipy.io as scio

###### map for object index and name #######
ycb_object = {
1: '002_master_chef_can', 
2: '003_cracker_box',
3: '004_sugar_box',
4: '005_tomato_soup_can',
5: '006_mustard_bottle',
6: '007_tuna_fish_can',
7:  '008_pudding_box',
8:  '009_gelatin_box',
9:  '010_potted_meat_can', 
10:  '011_banana',
11:  '019_pitcher_base', 
12:  '021_bleach_cleanser', 
13:  '024_bowl', 
14:  '025_mug', 
15:  '035_power_drill',
16:  '036_wood_block', 
17:  '037_scissors', 
18:  '040_large_marker', 
19:  '051_large_clamp',
20:  '052_extra_large_clamp',
21:  '061_foam_brick'
}
############################################


def getRandomIndex(n, x):
    index = np.random.choice(np.arange(n), size=x, replace=False)
    return index

def generate_random_training_idx(total_data_num, training_num):
    ret = getRandomIndex(total_data_num,training_num)
    ret.sort()
    return ret

def generate_even_training_idx(total_data_num, interval,start_from = 0):
    ret = range(start_from, total_data_num, interval) 
    return ret

#Given meta file and object id, return its relative pose and camera parameters
def load_meta_ycb(meta_file,obj_index):
    data = scio.loadmat(meta_file)
    cls_indexes = data['cls_indexes']
    obj_i = np.where(cls_indexes == obj_index)[0][0]
    factor_depth = data['factor_depth'].item()
    r_cam = data['poses'][:,:,obj_i][0:3,0:3]
    t_cam = data['poses'][:,:,obj_i][:,3]
    K = data['intrinsic_matrix']
    return r_cam, t_cam, K,factor_depth

def load_bbox_ycb(bbox_file,obj_index):
    bbox = []
    with open(bbox_file, 'r') as f:
        lines = f.read().splitlines()
        for l in lines:
            res = list(map(str, l.split(' ')))
            if ycb_object[obj_index] == res[0]:
                bbox = [float(i) for i in res[1:]]
    return bbox

def read_from_ba(test_img_pose,num_observations,path):
    with open(path, 'r') as f:
        lines = f.read().splitlines()
        for l in lines:
            res = list(map(float, l.split(' ')))
            test_img_pose[res[0]]=(res[2:])
            num_observations[res[0]] = res[1]

def read_cov(test_img_cov,path):
    with open(path, 'r') as f:
        lines = f.read().splitlines()
        for l in lines:
            print(l)
            res = list(map(float, l.split(' ')))
            test_img_cov[res[0]]=(res[1:])
            
def load_info(path):
    with open(path, 'r') as f:
        info = yaml.load(f, Loader=yaml.CLoader)
        for eid in info.keys():
            if 'cam_K' in info[eid].keys():
                info[eid]['cam_K'] = np.array(info[eid]['cam_K']).reshape(
                    (3, 3))
            if 'cam_R_w2c' in info[eid].keys():
                info[eid]['cam_R_w2c'] = np.array(
                    info[eid]['cam_R_w2c']).reshape((3, 3))
            if 'cam_t_w2c' in info[eid].keys():
                info[eid]['cam_t_w2c'] = np.array(
                    info[eid]['cam_t_w2c']).reshape((3, 1))
    return info

def save_info(path, info):
    for im_id in sorted(info.keys()):
        im_info = info[im_id]
        if 'cam_K' in im_info.keys():
            im_info['cam_K'] = im_info['cam_K'].flatten().tolist()
        if 'cam_R_w2c' in im_info.keys():
            im_info['cam_R_w2c'] = im_info['cam_R_w2c'].flatten().tolist()
        if 'cam_t_w2c' in im_info.keys():
            im_info['cam_t_w2c'] = im_info['cam_t_w2c'].flatten().tolist()
    with open(path, 'w') as f:
        yaml.dump(info, f, Dumper=yaml.CDumper, width=10000)

def load_gt(path):
    with open(path, 'r') as f:
        gts = yaml.load(f, Loader=yaml.CLoader)
        for im_id, gts_im in gts.items():
            for gt in gts_im:
                if 'cam_R_m2c' in gt.keys():
                    gt['cam_R_m2c'] = np.array(gt['cam_R_m2c']).reshape((3, 3))
                if 'cam_t_m2c' in gt.keys():
                    gt['cam_t_m2c'] = np.array(gt['cam_t_m2c']).reshape((3, 1))
    return gts

def save_gt(path, gts):
    for im_id in sorted(gts.keys()):
        im_gts = gts[im_id]
        for gt in im_gts:
            if 'cam_R_m2c' in gt.keys():
                gt['cam_R_m2c'] = gt['cam_R_m2c'].flatten().tolist()
            if 'cam_t_m2c' in gt.keys():
                gt['cam_t_m2c'] = gt['cam_t_m2c'].flatten().tolist()
            if 'obj_bb' in gt.keys():
                gt['obj_bb'] = [int(x) for x in gt['obj_bb']]
    with open(path, 'w') as f:
        yaml.dump(gts, f, Dumper=yaml.CDumper, width=10000)

def gaussian_k(x0,y0,sigma, width, height):
    x = np.arange(0, width, 1, float) ## (width,)
    y = np.arange(0, height, 1, float)[:, np.newaxis] ## (height,1)
    return np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))
 
def generate_hm(height, width ,landmarks,s=3):
    Nlandmarks = landmarks.shape[0]
    hm = np.zeros((height, width, Nlandmarks), dtype = np.float32)
    for i in range(Nlandmarks):
        if not np.array_equal(landmarks[i], [-1,-1]):
            
            hm[:,:,i] = gaussian_k(landmarks[i][0],
                                    landmarks[i][1],
                                    s,width,height)
        else:
            hm[:,:,i] = np.zeros((height,width))
    return hm

