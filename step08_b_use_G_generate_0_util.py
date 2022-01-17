import numpy as np
import cv2

from step06_a_datas_obj import Range

import sys
sys.path.append("kong_util")
from build_dataset_combine import  method1

import matplotlib.pyplot as plt
######################################################################################################################################################################################################
def F_01_or_C_01_method1_visual_op(data):
    ''' data.shape 為 H, W, C， 不適 N, H, W, C 喔！ '''
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
    F_visual = (method1(x=data[..., x_ind], y=data[..., y_ind], mask=mask)[..., ::-1] * 255.).astype(np.uint8)
    Cx_visual = (data[..., x_ind] * 255).astype(np.uint8)
    Cy_visual = (data[..., y_ind] * 255).astype(np.uint8)
    return F_visual, Cx_visual, Cy_visual
######################################################################################################################################################################################################
######################################################################################################################################################################################################
def Value_Range_Postprocess_to_01(data_pre, use_gt_range=Range( 0, 1)):
    if  (use_gt_range == Range(-1, 1)): data = (data_pre + 1) / 2   ### 如果 use_gt_range 是 -1~1 記得轉回 0~1
    elif(use_gt_range == Range( 0, 1)): data = data_pre
    return data
######################################################################################################################################################################################################
def C_01_concat_with_M_to_F_and_get_F_visual(C, M):
    F        = np.concatenate([M, C], axis=-1)  ### channel concate
    F_visual, Cx_visual, Cy_visual = F_01_or_C_01_method1_visual_op(F)
    return F, F_visual, Cx_visual, Cy_visual

def C_01_and_C_01_w_M_to_F_and_visualize(C, M):
    F, F_visual, Cx_visual, Cy_visual = C_01_concat_with_M_to_F_and_get_F_visual(C, M)

    C_w_M = C * M
    F_w_M, F_w_M_visual, Cx_w_M_visual, Cy_w_M_visual = C_01_concat_with_M_to_F_and_get_F_visual(C_w_M, M)
    return F, F_visual, Cx_visual, Cy_visual, F_w_M, F_w_M_visual, Cx_w_M_visual, Cy_w_M_visual

def W_01_visual_op(W_01):
    W_visual  = (W_01[..., 0:3] * 255).astype(np.uint8)
    Wz_visual = (W_01[..., 0:1] * 255).astype(np.uint8)
    Wy_visual = (W_01[..., 1:2] * 255).astype(np.uint8)
    Wx_visual = (W_01[..., 2:3] * 255).astype(np.uint8)
    return W_visual, Wx_visual, Wy_visual, Wz_visual
