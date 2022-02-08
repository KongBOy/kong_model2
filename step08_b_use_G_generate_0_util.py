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
    Cx_visual = (data[..., x_ind:x_ind + 1] * 255).astype(np.uint8)
    Cy_visual = (data[..., y_ind:y_ind + 1] * 255).astype(np.uint8)
    return F_visual, Cx_visual, Cy_visual
######################################################################################################################################################################################################
######################################################################################################################################################################################################
def Value_Range_Postprocess_to_01(data_pre, use_gt_range=Range( 0, 1)):
    if  (use_gt_range == Range(-1, 1)): data = (data_pre + 1) / 2   ### 如果 use_gt_range 是 -1~1 記得轉回 0~1
    elif(use_gt_range == Range( 0, 1)): data = data_pre
    return data
######################################################################################################################################################################################################
def C_01_concat_with_M_to_F_and_get_F_visual(C, M):
    ''' 輸入的 Coord 和 Mask 皆為 HWC '''
    h, w, c = C.shape  ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
    M = M[:h, :w, :]   ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
    F        = np.concatenate([M, C], axis=-1)  ### channel concate
    F_visual, Cx_visual, Cy_visual = F_01_or_C_01_method1_visual_op(F)
    return F, F_visual, Cx_visual, Cy_visual

def C_01_and_C_01_w_M_to_F_and_visualize(C, M):
    h, w, c = C.shape  ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
    M = M[:h, :w, :]   ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
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

def W_01_concat_with_M_to_WM_and_get_W_visual(W, M):
    ''' 輸入的 W 和 Mask 皆為 HWC '''
    h, w, c = W.shape  ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
    M = M[:h, :w, :]   ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
    WM = np.concatenate([W, M], axis=-1)
    W_visual, Wx_visual, Wy_visual, Wz_visual = W_01_visual_op(W)
    return WM, W_visual, Wx_visual, Wy_visual, Wz_visual

def W_01_and_W_01_w_M_to_WM_and_visualize(W_raw, M):
    W_raw_c_M, W_raw_visual, Wx_raw_visual, Wy_raw_visual, Wz_raw_visual = W_01_concat_with_M_to_WM_and_get_W_visual(W_raw, M)

    W_w_M = W_raw * M
    W_w_M_c_M, W_w_M_visual, Wx_w_M_visual, Wy_w_M_visual, Wz_w_M_visual = W_01_concat_with_M_to_WM_and_get_W_visual(W_w_M, M)
    return W_raw_c_M, W_raw_visual, Wx_raw_visual, Wy_raw_visual, Wz_raw_visual, W_w_M_c_M, W_w_M_visual, Wx_w_M_visual, Wy_w_M_visual, Wz_w_M_visual
