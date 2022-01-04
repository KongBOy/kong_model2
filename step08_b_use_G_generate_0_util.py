import numpy as np
import cv2

from step06_a_datas_obj import Range

import sys
sys.path.append("kong_util")
from build_dataset_combine import  method1

import matplotlib.pyplot as plt
######################################################################################################################################################################################################
def flow_or_coord_visual_op(data):
    data_ch = data.shape[2]
    mask = None
    x_ind = 0
    y_ind = 0
    if  (data_ch == 3):
        '''
        mask: mask/y/x
        '''
        # mask = data[..., 0]  ### 因為想看有沒有外溢出去所以丟None
        mask = None
        x_ind = 2
        y_ind = 1
    elif(data_ch == 2):
        '''
        coord: y/x
        '''
        mask = None
        x_ind = 1
        y_ind = 0
    return (method1(x=data[..., x_ind], y=data[..., y_ind], mask=mask)[..., ::-1] * 255.).astype(np.uint8)
######################################################################################################################################################################################################
######################################################################################################################################################################################################
def Value_Range_Postprocess_to_01(data_pre, use_gt_range=Range( 0, 1)):
    if  (use_gt_range == Range(-1, 1)): data = (data_pre + 1) / 2   ### 如果 use_gt_range 是 -1~1 記得轉回 0~1
    elif(use_gt_range == Range( 0, 1)): data = data_pre
    return data
######################################################################################################################################################################################################
def C_with_M_to_F_and_get_F_visual(coord, mask):
    flow        = np.concatenate([mask, coord], axis=-1)  ### channel concate
    flow_visual = flow_or_coord_visual_op(flow)
    return flow, flow_visual

def W_01_visual_op(W_01):
    W_visual  = (W_01[..., 0:3] * 255).astype(np.uint8)
    Wz_visual = (W_01[..., 0:1] * 255).astype(np.uint8)
    Wy_visual = (W_01[..., 1:2] * 255).astype(np.uint8)
    Wx_visual = (W_01[..., 2:3] * 255).astype(np.uint8)
    return W_visual, Wx_visual, Wy_visual, Wz_visual