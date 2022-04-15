import numpy as np
import cv2

from step06_a_datas_obj import Range

import sys
sys.path.append("kong_util")
from build_dataset_combine import  method1

import matplotlib.pyplot as plt
######################################################################################################################################################################################################
def F_01_or_C_01_method1_visual_op(data, out_ch3=False):
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
    if(out_ch3):
        Cx_visual = np.tile(Cx_visual, (1, 1, 3))
        Cy_visual = np.tile(Cy_visual, (1, 1, 3))
    return F_visual, Cx_visual, Cy_visual
######################################################################################################################################################################################################
######################################################################################################################################################################################################
def Value_Range_Postprocess_to_01(data_pre, use_gt_range=Range( 0, 1)):
    if  (use_gt_range == Range(-1, 1)): data = (data_pre + 1) / 2   ### 如果 use_gt_range 是 -1~1 記得轉回 0~1
    elif(use_gt_range == Range( 0, 1)): data = data_pre
    return data
######################################################################################################################################################################################################
def C_01_concat_with_M_to_F_and_get_F_visual(C, M, out_ch3=False):
    ''' 輸入的 Coord 和 Mask 皆為 HWC '''
    h, w, c = C.shape  ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
    M = M[:h, :w, :]   ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
    F        = np.concatenate([M, C], axis=-1)  ### channel concate
    F_visual, Cx_visual, Cy_visual = F_01_or_C_01_method1_visual_op(F, out_ch3=out_ch3)
    return F, F_visual, Cx_visual, Cy_visual

def C_01_and_C_01_w_M_to_F_and_visualize(C, M, out_ch3=False):
    h, w, c = C.shape  ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
    M = M[:h, :w, :]   ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
    F, F_visual, Cx_visual, Cy_visual = C_01_concat_with_M_to_F_and_get_F_visual(C, M, out_ch3=out_ch3)

    C_w_M = C * M
    F_w_M, F_w_M_visual, Cx_w_M_visual, Cy_w_M_visual = C_01_concat_with_M_to_F_and_get_F_visual(C_w_M, M, out_ch3=out_ch3)
    return F, F_visual, Cx_visual, Cy_visual, F_w_M, F_w_M_visual, Cx_w_M_visual, Cy_w_M_visual

def W_01_visual_op(W_01, out_ch3=False):
    W_visual  = (W_01[..., 0:3] * 255).astype(np.uint8)
    Wz_visual = (W_01[..., 0:1] * 255).astype(np.uint8)
    Wy_visual = (W_01[..., 1:2] * 255).astype(np.uint8)
    Wx_visual = (W_01[..., 2:3] * 255).astype(np.uint8)
    if(out_ch3):
        Wx_visual = np.tile(Wx_visual, (1, 1, 3))
        Wy_visual = np.tile(Wy_visual, (1, 1, 3))
        Wz_visual = np.tile(Wz_visual, (1, 1, 3))
    return W_visual, Wx_visual, Wy_visual, Wz_visual

def W_01_concat_with_M_to_WM_and_get_W_visual(W, M, out_ch3=False):
    ''' 輸入的 W 和 Mask 皆為 HWC '''
    h, w, c = W.shape  ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
    M = M[:h, :w, :]   ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
    WM = np.concatenate([W, M], axis=-1)
    W_visual, Wx_visual, Wy_visual, Wz_visual = W_01_visual_op(W, out_ch3=out_ch3)
    return WM, W_visual, Wx_visual, Wy_visual, Wz_visual

def W_01_and_W_01_w_M_to_WM_and_visualize(W_raw, M, out_ch3=False):
    W_raw_c_M, W_raw_visual, Wx_raw_visual, Wy_raw_visual, Wz_raw_visual = W_01_concat_with_M_to_WM_and_get_W_visual(W_raw, M, out_ch3=out_ch3)
    h, w, c = W_raw.shape  ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
    M = M[:h, :w, :]       ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
    W_w_M = W_raw * M
    W_w_M_c_M, W_w_M_visual, Wx_w_M_visual, Wy_w_M_visual, Wz_w_M_visual = W_01_concat_with_M_to_WM_and_get_W_visual(W_w_M, M, out_ch3=out_ch3)
    return W_raw_c_M, W_raw_visual, Wx_raw_visual, Wy_raw_visual, Wz_raw_visual, W_w_M_c_M, W_w_M_visual, Wx_w_M_visual, Wy_w_M_visual, Wz_w_M_visual

######################################################################################################################################################################################################
######################################################################################################################################################################################################
import tensorflow as tf
class Tight_crop():
    def __init__(self, pad_size=20, resize=None, jit_scale=0):
        self.pad_size  = pad_size
        self.resize    = resize
        self.jit_scale = jit_scale

        ### tf.random.uniform 不能夠接受 jit_scale = 0， 所以才必須要有這個if
        if(self.jit_scale > 0):  self.reset_jit()

    def reset_jit(self):
        if(self.jit_scale > 0):
            self.l_jit = tf.random.uniform(shape=[], minval=-self.jit_scale, maxval=self.jit_scale, dtype=tf.int64)
            self.r_jit = tf.random.uniform(shape=[], minval=-self.jit_scale, maxval=self.jit_scale, dtype=tf.int64)
            self.t_jit = tf.random.uniform(shape=[], minval=-self.jit_scale, maxval=self.jit_scale, dtype=tf.int64)
            self.d_jit = tf.random.uniform(shape=[], minval=-self.jit_scale, maxval=self.jit_scale, dtype=tf.int64)
        else: print("Tight_crop=0 無法 reset_jit， 請注意 建立 Tight_crop物件 的時候 jit_scale 有沒有設定數值， 目前不做事直接跳過 reset_jit動作")

    def __call__(self, data, Mask):
        '''
        目前的寫法連 batch 都考慮進去囉
        resize: [h, w]
        '''
        ### np.where( T/F_map 或 0/非0_array ) 可參考：https://numpy.org/doc/stable/reference/generated/numpy.where.html，
        ### np.where( T/F_map 或 0/非0_array ) 只有放一個參數的時候， 相當於 np.nonzero()， 如果放三個參數時， 為true的地方 填入 第二個參數值， 為False的地方 填入 第三個參數值,
        ### 不直接用 nonzero的原因是 tf 沒有 nonzero 但有 where， 為了 tf, numpy 都通用 這邊 numpy 就配合tf 用 where 囉～
        ### 但也要注意 tf.where( T/F_map 或 0/非0_array ) return 回來的東西 跟 np.where( T/F_map 或 0/非0_array ) 回傳的東西 shape 不大一樣喔
        ### np.where( T/F_map 或 0/非0_array ) -> tuple ( x非零indexs, y非零indexs, z非零indexs )
        ### tf.where( T/F_map 或 0/非0_array ) -> tensor.shape( 所有非零點的個數, 3 ( 非零xyz_index) )
        nonzero_map = Mask > 0
        ### numpy 寫法：
        # if  (len(Mask.shape) == 4): b_ind, y_ind, x_ind, c_ind = np.where(nonzero_map)
        # elif(len(Mask.shape) == 3): y_ind, x_ind, c_ind = np.where(nonzero_map)
        # elif(len(Mask.shape) == 2): y_ind, x_ind = np.where(nonzero_map)

        ### tf 寫法
        nonzero_index = tf.where(nonzero_map)
        if  (len(Mask.shape) == 4): col_id = 1  ### BHWC， H在第1個col
        elif(len(Mask.shape) == 3): col_id = 0  ### HWC ， H在第0個col
        elif(len(Mask.shape) == 2): col_id = 0  ### HWC ， H在第0個col
        y_ind = nonzero_index[:, col_id     : col_id + 1]
        x_ind = nonzero_index[:, col_id + 1 : col_id + 2]

        # l = x_ind.min()
        # r = x_ind.max()
        # t = y_ind.min()
        # d = y_ind.max()
        l = tf.reduce_min(x_ind)
        r = tf.reduce_max(x_ind)
        t = tf.reduce_min(y_ind)
        d = tf.reduce_max(y_ind)

        l_pad = l - self.pad_size
        r_pad = r + self.pad_size
        t_pad = t - self.pad_size
        d_pad = d + self.pad_size

        ### 隨機抖動 random jit
        if(self.jit_scale > 0):  ### tf.random.uniform 不能夠接受 jit_scale = 0， 所以才必須要有這個if
            l_pad += self.l_jit
            r_pad += self.r_jit
            t_pad += self.t_jit
            d_pad += self.d_jit

        ########### 先 pad 再 crop
        ###### pad part
        ### 看 超過影像範圍多少
        l_out = tf.constant(0, tf.int64)
        r_out = tf.constant(0, tf.int64)
        t_out = tf.constant(0, tf.int64)
        d_out = tf.constant(0, tf.int64)
        if  (len(data.shape) == 4): b, h, w, c = data.shape
        elif(len(data.shape) == 3): h, w, c = data.shape
        elif(len(data.shape) == 2): h, w = data.shape

        if(l_pad < 0): l_out = - l_pad
        if(t_pad < 0): t_out = - t_pad
        if(r_pad > w - 1): r_out = r_pad - (w - 1)
        if(d_pad > h - 1): d_out = d_pad - (h - 1)

        ### 看 pad 的範圍有沒有超過影像， 有的話就 pad
        if(l_out > 0 or r_out > 0  or t_out > 0 or d_out > 0):
            # if  (len(data.shape) == 4): data = np.pad(data, ( (0    ,     0), (t_out, d_out), (l_out, r_out), (    0,     0) ) , 'reflect')
            # elif(len(data.shape) == 3): data = np.pad(data, ( (t_out, d_out), (l_out, r_out), (    0,     0) )                 , 'reflect')
            # elif(len(data.shape) == 2): data = np.pad(data, ( (t_out, d_out), (l_out, r_out) )                                 , 'reflect')
            if  (len(data.shape) == 4): data = tf.pad(data, ( (0    ,     0), (t_out, d_out), (l_out, r_out), (    0,     0) ) , 'REFLECT')
            elif(len(data.shape) == 3): data = tf.pad(data, ( (t_out, d_out), (l_out, r_out), (    0,     0) )                 , 'REFLECT')
            elif(len(data.shape) == 2): data = tf.pad(data, ( (t_out, d_out), (l_out, r_out) )                                 , 'REFLECT')
        # breakpoint()

        ###### pad 完成了， 以下開始 crop
        ### 對 pad完成 的 data 重新定位
        if(l_pad < 0): l_pad = tf.constant(0, tf.int64)
        if(t_pad < 0): t_pad = tf.constant(0, tf.int64)
        # l_pad = max(l_pad, 0)          ### l_pad, t_pad 可能會被剪到 負的， 但index最小是0喔 ， 所以最小取0
        # t_pad = max(t_pad, 0)          ### l_pad, t_pad 可能會被剪到 負的， 但index最小是0喔 ， 所以最小取0
        r_pad = r_pad + l_out + r_out  ### r_pad, d_pad 自己如果超過的話， 因為會pad出去， 所以要加上 超過的部分， 在來還要考慮如果 l_pad, t_pad 超出去的話， 因為index最小為0， 代表 左、上 超出去的部分 要補到 右、下 的部分， 所以要多加 l_out, t_out 喔！
        d_pad = d_pad + t_out + d_out  ### r_pad, d_pad 自己如果超過的話， 因為會pad出去， 所以要加上 超過的部分， 在來還要考慮如果 l_pad, t_pad 超出去的話， 因為index最小為0， 代表 左、上 超出去的部分 要補到 右、下 的部分， 所以要多加 l_out, t_out 喔！

        ### 重新定位 完成了， 以下開始 crop
        if  (len(data.shape) == 4): data = data[:, t_pad : d_pad + 1, l_pad : r_pad + 1, :]  ### BHWC
        elif(len(data.shape) == 3): data = data[t_pad : d_pad + 1, l_pad : r_pad + 1, :]     ### HWC
        elif(len(data.shape) == 2): data = data[t_pad : d_pad + 1, l_pad : r_pad + 1]        ### HW

        ########### 全都處理完以後， resize 到指定的大小
        if(self.resize is not None): data = tf.image.resize(data, self.resize, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # breakpoint()
        return data

######################################################################################################################################################################################################
######################################################################################################################################################################################################
class Use_G_generate:
    def __init__(self):
        self.model_obj      = None
        self.phase          = None
        self.index          = None
        self.in_ord         = None
        self.in_pre         = None
        self.gt_ord         = None
        self.gt_pre         = None
        self.rec_hope       = None
        self.exp_obj        = None
        self.training       = None
        self.see_reset_init = None
        self.postprocess    = None
        self.npz_save       = None
        self.add_loss       = None
        self.bgr2rgb        = None

    def __call__(self, model_obj, phase, index, in_ord, in_pre, gt_ord, gt_pre, rec_hope=None, exp_obj=None, training=True, see_reset_init=True, postprocess=False, npz_save=False, add_loss=False, bgr2rgb=True):
        self.model_obj      = model_obj
        self.phase          = phase
        self.index          = index
        self.in_ord         = in_ord
        self.in_pre         = in_pre
        self.gt_ord         = gt_ord
        self.gt_pre         = gt_pre
        self.rec_hope       = rec_hope
        self.exp_obj        = exp_obj
        self.training       = training
        self.see_reset_init = see_reset_init
        self.postprocess    = postprocess
        self.npz_save       = npz_save
        self.add_loss       = add_loss
        self.bgr2rgb        = bgr2rgb
        self.doing_things()

    def doing_things(self):
        pass

# class Use_G_generate:
#     def __call__(self,model_obj, phase, index, in_ord, in_pre, gt_ord, gt_pre, rec_hope=None, exp_obj=None, training=True, see_reset_init=True, postprocess=False, npz_save=False, add_loss=False, bgr2rgb=False):
#         self.doing_things(model_obj, phase, index, in_ord, in_pre, gt_ord, gt_pre, rec_hope=None, exp_obj=None, training=True, see_reset_init=True, postprocess=False, npz_save=False, add_loss=False, bgr2rgb=False)

#     def doing_things(self,model_obj, phase, index, in_ord, in_pre, gt_ord, gt_pre, rec_hope=None, exp_obj=None, training=True, see_reset_init=True, postprocess=False, npz_save=False, add_loss=False, bgr2rgb=False):
#         ''' Not Implement'''
#         pass
