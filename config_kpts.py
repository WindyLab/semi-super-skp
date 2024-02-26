import torch

###### global parameters #######
iter_rd = 2     ### starts from 1, change this to record dual optimization cycle
num_classes = 9 ### number of key points classes

config_drone = {
    'data_name': 'phantom',
    'device': torch.device("cuda"),
    'num_classes': num_classes,
    'batch_size': 32,
    'scale': 6,
    'val_percent': 0,
    'img_h': 2160,
    'img_w': 3840,
    'iter_rd': iter_rd,
    'load_pretrained': False,
    'img_dir': '/media/datadisk/data_space/',
    'pretrained_path': 'checkpoints/Model_Ep280.pth',
    'label_dir': 'labels',
    'checkpoint_dir': 'checkpoints'
}

################ 3D key points for projection #################
### 004_sugar_box
pts = [ 
        [ -0.0213,-0.0429,0.0824],
        [-0.01944 ,0.043467 ,0.081095 ],
        [ 0.016658 ,0.0419 ,0.082 ],
        [0.012622,-0.043976,0.081258],
        [-0.013117,-0.04739,-0.089 ],
        [-0.0065,0.044122,-0.088209],
        [0.0249 ,0.039208,-0.088782  ],
        [0.021157 ,-0.0468 ,-0.088193 ]
]
###############################################################

config_ycb = {
    'data_name': 'ycb',
    'data_set_ids':[25],
    'obj_index': 3,
    'device': torch.device("cuda"),
    'num_classes': num_classes,
    'pts3d': pts,
    'val_percent': 0,
    'img_h': 240,
    'img_w': 320,
    'iter_rd': iter_rd,
    'load_pretrained': False,
    'img_dir': '/media/datadisk/data_space/YCB_Video_Dataset/data/',
    'pretrained_path': 'checkpoints/Model_Ep280.pth',
    'label_dir': 'labels',
    'checkpoint_dir': 'checkpoints'
}