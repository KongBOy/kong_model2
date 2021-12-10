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
def F_postprocess(flow_pre, use_gt_range):
    if  (use_gt_range == Range(-1, 1)): flow = (flow_pre + 1) / 2   ### 如果 use_gt_range 是 -1~1 記得轉回 0~1
    elif(use_gt_range == Range( 0, 1)): flow = flow_pre
    # flow [..., 1] = 1 - flow[..., 1]  ### y 上下 flip， 雖然背景會變成青色， 不過就試試看囉， 算了好麻煩還是保持原樣：在視覺化的時候 先不要 y_flip， 在rec時再flip好了～
    # flow = flow[..., 0:1] * flow      ### 因為想看 pred_C 有沒有外溢， 所以就先不跟mask 相乘
    return flow
######################################################################################################################################################################################################
def C_postprocess(coord_pre, use_gt_range):
    if  (use_gt_range == Range(-1, 1)): coord = (coord_pre + 1) / 2   ### 如果 use_gt_range 是 -1~1 記得轉回 0~1
    elif(use_gt_range == Range( 0, 1)): coord = coord_pre
    # coord [..., 0] = 1 - coord[..., 0]  ### y 上下 flip， 雖然背景會變成青色， 不過就試試看囉， 算了好麻煩還是保持原樣：在視覺化的時候 先不要 y_flip， 在rec時再flip好了～
    return coord
######################################################################################################################################################################################################
def C_with_M_to_F_and_get_F_visual(coord, mask):
    flow        = np.concatenate([mask, coord], axis=-1)  ### channel concate
    flow_visual = flow_or_coord_visual_op(flow)
    return flow, flow_visual