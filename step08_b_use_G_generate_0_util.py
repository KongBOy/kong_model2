import numpy as np
import cv2

from step06_a_datas_obj import Range

import sys
sys.path.append("kong_util")
from kong_util.build_dataset_combine import method1, Check_dir_exist_and_build, Save_npy_path_as_knpy

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

def WcM_01_visual_op(WcM_01, out_ch3=False):
    ''' W_concat_M 的縮寫， 前三ch W， 第四ch為M'''
    W_visual  = (WcM_01[..., 0:3] * 255).astype(np.uint8)
    Wz_visual = (WcM_01[..., 0:1] * 255).astype(np.uint8)
    Wy_visual = (WcM_01[..., 1:2] * 255).astype(np.uint8)
    Wx_visual = (WcM_01[..., 2:3] * 255).astype(np.uint8)
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
    W_visual, Wx_visual, Wy_visual, Wz_visual = WcM_01_visual_op(W, out_ch3=out_ch3)
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
class Tight_crop():
    def __init__(self, pad_size=20, pad_method="reflect+black", resize=None, jit_scale=0):
        '''
        pad_method：
            reflect all  ： 最真實， 但比較慢 ( 覺得最後精細train的時候再用這個， 現在先用比較快的方法吧 )
            reflect+black： 已經算滿真實了
        '''
        self.pad_size   = pad_size
        self.pad_method = pad_method
        self.resize    = resize
        self.jit_scale = jit_scale

        ### tf.random.uniform 不能夠接受 jit_scale = 0， 所以才必須要有這個if
        if(self.jit_scale > 0):  self.reset_jit()

    def reset_jit(self):
        import tensorflow as tf
        if(self.jit_scale > 0):
            self.l_jit = tf.random.uniform(shape=[], minval=-self.jit_scale, maxval=self.jit_scale, dtype=tf.int64)
            self.r_jit = tf.random.uniform(shape=[], minval=-self.jit_scale, maxval=self.jit_scale, dtype=tf.int64)
            self.t_jit = tf.random.uniform(shape=[], minval=-self.jit_scale, maxval=self.jit_scale, dtype=tf.int64)
            self.d_jit = tf.random.uniform(shape=[], minval=-self.jit_scale, maxval=self.jit_scale, dtype=tf.int64)
        else: print("Tight_crop=0 無法 reset_jit， 請注意 建立 Tight_crop物件 的時候 jit_scale 有沒有設定數值， 目前不做事直接跳過 reset_jit動作")

    def reset_resize(self, resize):
        self.resize = resize

    def __call__(self, data, Mask):
        import tensorflow as tf
        if  (len(data.shape) == 4): b, h, w, c = data.shape
        elif(len(data.shape) == 3): h, w, c = data.shape
        elif(len(data.shape) == 2): h, w = data.shape

        # if( tf.math.equal(tf.reduce_sum(Mask), tf.constant(0, tf.float32) )):
        #     print("Mask 全黑， 不做 tight_crop")
        #     return data, {"l_pad_slice": tf.constant(0, tf.int64), "t_pad_slice": tf.constant(0, tf.int64), "r_pad_slice": tf.constant(w, tf.int64), "d_pad_slice": tf.constant(h, tf.int64)}

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
        l = tf.reduce_min(x_ind)  ### index
        r = tf.reduce_max(x_ind)  ### index
        t = tf.reduce_min(y_ind)  ### index
        d = tf.reduce_max(y_ind)  ### index

        l_pad_ind = l - self.pad_size  ### index
        r_pad_ind = r + self.pad_size  ### index
        t_pad_ind = t - self.pad_size  ### index
        d_pad_ind = d + self.pad_size  ### index

        ### 隨機抖動 random jit
        if(self.jit_scale > 0):  ### tf.random.uniform 不能夠接受 jit_scale = 0， 所以才必須要有這個if
            l_pad_ind += self.l_jit  ### index
            r_pad_ind += self.r_jit  ### index
            t_pad_ind += self.t_jit  ### index
            d_pad_ind += self.d_jit  ### index

        ########### 先 pad 再 crop
        ###### pad part
        ### 看 超過影像範圍多少
        l_out_amo = tf.constant(0, tf.int64)  ### 格數
        r_out_amo = tf.constant(0, tf.int64)  ### 格數
        t_out_amo = tf.constant(0, tf.int64)  ### 格數
        d_out_amo = tf.constant(0, tf.int64)  ### 格數

        if(l_pad_ind < 0): l_out_amo = - l_pad_ind  ### l, t 負的 index 剛好 == 超出去的格數
        if(t_pad_ind < 0): t_out_amo = - t_pad_ind  ### l, t 負的 index 剛好 == 超出去的格數
        if(r_pad_ind > (w - 1) ): r_out_amo = r_pad_ind - (w - 1)  ### -1 是 格數 轉 index， index - index 後 就是 超出去的格數囉
        if(d_pad_ind > (h - 1) ): d_out_amo = d_pad_ind - (h - 1)  ### -1 是 格數 轉 index， index - index 後 就是 超出去的格數囉

        ### 看 pad 的範圍有沒有超過影像， 有的話就 pad
        if(l_out_amo > 0 or r_out_amo > 0  or t_out_amo > 0 or d_out_amo > 0):
            if(self.pad_method == "reflect+black"):
                ''' 
                想用 reflect 的 padding 比較接近真實，
                但是要注意！太超過不能用 reflect 的 padding 喔！
                因為 mask 也會被reflect過去，
                所以：
                    1. 超出在 頁面邊界 到 圖邊界 之間的範圍 用 reflect padding
                    2. 更超過 用 black padding 這樣子拉
                '''
                ### 計算 頁面邊界 到 圖邊界的格數囉
                l_to_board_amo = l  ### l, t index 剛好就是 頁面邊界 到 圖邊界的格數囉
                t_to_board_amo = t  ### l, t index 剛好就是 頁面邊界 到 圖邊界的格數囉
                r_to_board_amo = (w - 1) - r  ### -1 是 格數 轉 index， w/h_index - r/d_index 後 就是 r, d 的 頁面邊界 到 圖邊界的格數囉
                d_to_board_amo = (h - 1) - d  ### -1 是 格數 轉 index， w/h_index - r/d_index 後 就是 r, d 的 頁面邊界 到 圖邊界的格數囉

                ### 計算 reflect_pad 的格數 ( 即to_board格數， 但別忘記是要在 out 的情況下 才有需要 pad 喔！ )
                l_reflect_amo = tf.constant(0, tf.int64)
                t_reflect_amo = tf.constant(0, tf.int64)
                r_reflect_amo = tf.constant(0, tf.int64)
                d_reflect_amo = tf.constant(0, tf.int64)
                if( l_out_amo > 0): l_reflect_amo = tf.math.minimum(l_out_amo, l_to_board_amo) - 1  ### 扣除board自己本身， 因為reflect是 board該格本身開始做 reflect， 所以 圖邊界 到 頁面邊界 之間 可用的空間就少一個囉， 少 board那格！
                if( t_out_amo > 0): t_reflect_amo = tf.math.minimum(t_out_amo, t_to_board_amo) - 1  ### 扣除board自己本身， 因為reflect是 board該格本身開始做 reflect， 所以 圖邊界 到 頁面邊界 之間 可用的空間就少一個囉， 少 board那格！
                if( r_out_amo > 0): r_reflect_amo = tf.math.minimum(r_out_amo, r_to_board_amo) - 1  ### 扣除board自己本身， 因為reflect是 board該格本身開始做 reflect， 所以 圖邊界 到 頁面邊界 之間 可用的空間就少一個囉， 少 board那格！
                if( d_out_amo > 0): d_reflect_amo = tf.math.minimum(d_out_amo, d_to_board_amo) - 1  ### 扣除board自己本身， 因為reflect是 board該格本身開始做 reflect， 所以 圖邊界 到 頁面邊界 之間 可用的空間就少一個囉， 少 board那格！
                if  (len(data.shape) == 4): data = tf.pad(data, ( (0    ,     0), (t_reflect_amo, d_reflect_amo), (l_reflect_amo, r_reflect_amo), (    0,     0) ) , 'REFLECT')
                elif(len(data.shape) == 3): data = tf.pad(data, ( (t_reflect_amo, d_reflect_amo), (l_reflect_amo, r_reflect_amo), (    0,     0) )                 , 'REFLECT')
                elif(len(data.shape) == 2): data = tf.pad(data, ( (t_reflect_amo, d_reflect_amo), (l_reflect_amo, r_reflect_amo) )                                 , 'REFLECT')

                ### 計算 剩下還需要pad多少黑邊格數( out格數 - reflect格數)
                l_black_amo = tf.constant(0, tf.int64)
                t_black_amo = tf.constant(0, tf.int64)
                r_black_amo = tf.constant(0, tf.int64)
                d_black_amo = tf.constant(0, tf.int64)
                if(l_out_amo > l_to_board_amo): l_black_amo = l_out_amo - l_reflect_amo
                if(t_out_amo > t_to_board_amo): t_black_amo = t_out_amo - t_reflect_amo
                if(r_out_amo > r_to_board_amo): r_black_amo = r_out_amo - r_reflect_amo
                if(d_out_amo > d_to_board_amo): d_black_amo = d_out_amo - d_reflect_amo
                if  (len(data.shape) == 4): data = tf.pad(data, ( (0    ,     0), (t_black_amo, d_black_amo), (l_black_amo, r_black_amo), (    0,     0) ) , 'CONSTANT')
                elif(len(data.shape) == 3): data = tf.pad(data, ( (t_black_amo, d_black_amo), (l_black_amo, r_black_amo), (    0,     0) )                 , 'CONSTANT')
                elif(len(data.shape) == 2): data = tf.pad(data, ( (t_black_amo, d_black_amo), (l_black_amo, r_black_amo) )                                 , 'CONSTANT')

            elif(self.pad_method == "reflect all"):
                ''' 
                想用 reflect 的 padding 比較接近真實，
                但是要注意！太超過不能用 reflect 的 padding 喔！
                因為 mask 也會被reflect過去，
                所以：
                    1. 超出在 頁面邊界 到 圖邊界 之間的範圍 用 reflect padding
                    2. 如果還有更超過的， 用 1. 的結果 再次 reflect padding， 反覆迭代直到 剩下需要pad 的格數為0為止
                '''
                ### 剩下多少的格數 需要做 pad
                l_remian_amo = l_out_amo
                t_remian_amo = t_out_amo
                r_remian_amo = r_out_amo
                d_remian_amo = d_out_amo

                ### 計算 頁面邊界 到 圖邊界的格數囉
                l_to_board_amo = l  ### l, t index 剛好就是 頁面邊界 到 圖邊界的格數囉
                t_to_board_amo = t  ### l, t index 剛好就是 頁面邊界 到 圖邊界的格數囉
                r_to_board_amo = (w - 1) - r  ### -1 是 格數 轉 index， w/h_index - r/d_index 後 就是 r, d 的 頁面邊界 到 圖邊界的格數囉
                d_to_board_amo = (h - 1) - d  ### -1 是 格數 轉 index， w/h_index - r/d_index 後 就是 r, d 的 頁面邊界 到 圖邊界的格數囉

                ### 反覆迭代直到 剩下需要pad 的格數為0為止
                while( l_remian_amo > 0 or t_remian_amo > 0 or r_remian_amo > 0 or d_remian_amo > 0):
                    tf.autograph.experimental.set_loop_options(maximum_iterations=5,
                                                               shape_invariants=[(data, tf.TensorShape([None, None, None, None]))])  ### 參考：https://www.tensorflow.org/api_docs/python/tf/autograph/experimental/set_loop_options
                    ### 計算 reflect_pad 的格數 ( 別忘記是要在 out 的情況下 才有需要 pad 喔！ )， 最多pad的格數 只能用 頁面邊界~圖邊界的格數喔！ 所以就看 要補的格數 和 頁面邊界~圖邊界 的格數 哪個小， 就用哪個來reflect pad
                    l_reflect_amo = tf.constant(0, tf.int64)
                    t_reflect_amo = tf.constant(0, tf.int64)
                    r_reflect_amo = tf.constant(0, tf.int64)
                    d_reflect_amo = tf.constant(0, tf.int64)
                    if(l_remian_amo > 0):  ###別忘記是要在 out 的情況下 才有需要 pad 喔！ l, t, r, d 同理， 註解就不重複打了
                        ### 檢查 要pad的格數 是否會觸碰到 頁面邊界～                      l, t, r, d 同理， 註解就不重複打了
                        if(l_remian_amo >= l_to_board_amo): l_reflect_amo = l_to_board_amo - 1  ### 扣除board自己本身， 因為reflect是 board該格本身開始做 reflect， 所以 圖邊界 到 頁面邊界 之間 可用的空間就少一個囉， 少 board那格！
                        else                              : l_reflect_amo = l_remian_amo        ### 不會有碰觸到 邊界的問題， 放心的把 remain的格數 pad完吧
                    if(t_remian_amo > 0):
                        if(t_remian_amo >= t_to_board_amo): t_reflect_amo = t_to_board_amo - 1  ### 扣除board自己本身， 因為reflect是 board該格本身開始做 reflect， 所以 圖邊界 到 頁面邊界 之間 可用的空間就少一個囉， 少 board那格！
                        else                              : t_reflect_amo = t_remian_amo        ### 不會有碰觸到 邊界的問題， 放心的把 remain的格數 pad完吧
                    if(r_remian_amo > 0):
                        if(r_remian_amo >= r_to_board_amo): r_reflect_amo = r_to_board_amo - 1  ### 扣除board自己本身， 因為reflect是 board該格本身開始做 reflect， 所以 圖邊界 到 頁面邊界 之間 可用的空間就少一個囉， 少 board那格！
                        else                              : r_reflect_amo = r_remian_amo        ### 不會有碰觸到 邊界的問題， 放心的把 remain的格數 pad完吧
                    if(d_remian_amo > 0):
                        if(d_remian_amo >= d_to_board_amo): d_reflect_amo = d_to_board_amo - 1  ### 扣除board自己本身， 因為reflect是 board該格本身開始做 reflect， 所以 圖邊界 到 頁面邊界 之間 可用的空間就少一個囉， 少 board那格！
                        else                              : d_reflect_amo = d_remian_amo        ### 不會有碰觸到 邊界的問題， 放心的把 remain的格數 pad完吧
                    ### 做 pad
                    if  (len(data.shape) == 4): data = tf.pad(data, ( (0    ,     0), (t_reflect_amo, d_reflect_amo), (l_reflect_amo, r_reflect_amo), (    0,     0) ) , 'REFLECT')
                    elif(len(data.shape) == 3): data = tf.pad(data, ( (t_reflect_amo, d_reflect_amo), (l_reflect_amo, r_reflect_amo), (    0,     0) )                 , 'REFLECT')
                    elif(len(data.shape) == 2): data = tf.pad(data, ( (t_reflect_amo, d_reflect_amo), (l_reflect_amo, r_reflect_amo) )                                 , 'REFLECT')

                    ### 計算 還有多少格數需要 pad
                    l_remian_amo -=  l_reflect_amo
                    t_remian_amo -=  t_reflect_amo
                    r_remian_amo -=  r_reflect_amo
                    d_remian_amo -=  d_reflect_amo

                    ### 如果 仍有要補的空間(remain_amo > 0)， 更新一下 to_board_amo 給下一輪用(因為 pad完後 會有更多的 to_board_amo 空間可以使用)
                    if(l_remian_amo > 0): l_to_board_amo += l_reflect_amo
                    if(t_remian_amo > 0): t_to_board_amo += t_reflect_amo
                    if(r_remian_amo > 0): r_to_board_amo += r_reflect_amo
                    if(d_remian_amo > 0): d_to_board_amo += d_reflect_amo
            # plt.imshow(data)  ### 看 mask_pre 的 crop結果 才準喔， 因為 mask_pre 是很明確的 0 跟 1 的數值， 不會像 dis_img 邊界pad時可能有有模糊的空間
            # plt.show()
        # breakpoint()

        ###### pad 完成了， 以下開始 crop
        ### 對 pad完成 的 data 重新定位
        # l_pad_ind = max(l_pad_ind, 0)  ### l_pad_ind, t_pad_ind 可能會被剪到 負的， 但index最小是0喔 ， 所以最小取0 ， 但位移後 多出來的 位置 就要加回去給 r_pad_ind, d_pad_ind 囉～
        # t_pad_ind = max(t_pad_ind, 0)  ### l_pad_ind, t_pad_ind 可能會被剪到 負的， 但index最小是0喔 ， 所以最小取0 ， 但位移後 多出來的 位置 就要加回去給 r_pad_ind, d_pad_ind 囉～
        if(l_pad_ind < 0):  ### l_pad_ind, t_pad_ind 可能會被剪到 負的， 但index最小是0喔 ， 所以最小取0， 但位移後 多出來的 位置 就要加回去給 r_pad_ind, d_pad_ind 囉～
            l_pad_ind = tf.constant(0, tf.int64)  ### tf.autograph 沒有辦法用 max()， 只好乖乖寫if囉
            r_pad_ind += l_out_amo                ### 多出來的 位置 就要加回去給 r_pad_ind, d_pad_ind 囉～
        if(t_pad_ind < 0):  ### l_pad_ind, t_pad_ind 可能會被剪到 負的， 但index最小是0喔 ， 所以最小取0， 但位移後 多出來的 位置 就要加回去給 r_pad_ind, d_pad_ind 囉～
            t_pad_ind = tf.constant(0, tf.int64)  ### tf.autograph 沒有辦法用 max()， 只好乖乖寫if囉
            d_pad_ind += t_out_amo               ### 多出來的 位置 就要加回去給 r_pad_ind, d_pad_ind 囉～

        ### 重新定位 完成了， 以下開始 crop
        l_pad_slice = l_pad_ind  ### l, t 的 index 剛好 == slice
        t_pad_slice = t_pad_ind  ### l, t 的 index 剛好 == slice
        d_pad_slice = d_pad_ind + 1  ### index 轉 slice
        r_pad_slice = r_pad_ind + 1  ### index 轉 slice
        if  (len(data.shape) == 4): data = data[:, t_pad_slice : d_pad_slice , l_pad_slice : r_pad_slice , :]  ### BHWC
        elif(len(data.shape) == 3): data = data   [t_pad_slice : d_pad_slice , l_pad_slice : r_pad_slice , :]  ### HWC
        elif(len(data.shape) == 2): data = data   [t_pad_slice : d_pad_slice , l_pad_slice : r_pad_slice]      ### HW

        ########### 全都處理完以後， resize 到指定的大小
        if(self.resize is not None): data = tf.image.resize(data, self.resize, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # breakpoint()
        return data, {"l_pad_slice": l_pad_slice, "t_pad_slice": t_pad_slice, "r_pad_slice": r_pad_slice, "d_pad_slice": d_pad_slice,
                      "l_out_amo"  : l_out_amo  , "t_out_amo"  : t_out_amo  , "r_out_amo"  : r_out_amo  , "d_out_amo"  : d_out_amo  }

    ### 要把 data_pre_croped_resized 的 結果 反向操作 回 data_pre
    def croped_back(self, cropped_resized_data, boundary, back_w, back_h):
        import tensorflow as tf
        l_out_amo = boundary["l_out_amo"]
        t_out_amo = boundary["t_out_amo"]
        r_out_amo = boundary["r_out_amo"]
        d_out_amo = boundary["d_out_amo"]
        l_pad_slice = boundary["l_pad_slice"]
        t_pad_slice = boundary["t_pad_slice"]
        r_pad_slice = boundary["r_pad_slice"]
        d_pad_slice = boundary["d_pad_slice"]

        ####### 從 croped_resize 回復到 croped 的大小
        croped_w = r_pad_slice - l_pad_slice
        croped_h = d_pad_slice - t_pad_slice
        cropped_data = tf.image.resize(cropped_resized_data, (croped_h, croped_w) )

        ####### 把 croped 的狀況有兩種， 1.一種是 超出img範圍往外pad， 2.一種是 img範圍內 往內crop
        ##### 1. 要把 往外pad  的部分切掉
        ##### 2. 要把 往內crop 的部分補回來
        ###  不能這樣寫， 因為譬如 l_out, r_out 同時 == 0， 會變 [0:-0]， 結果為空array， 但應該是要全取才對， 所以只好像下面 一步步展開來做
        # if  (len(cropped_data.shape) == 4): cropped_data = cropped_data[ :, t_out_amo : -d_out_amo,  boundary["l_out_amo"] : -r_out_amo, :]
        # elif(len(cropped_data.shape) == 3): cropped_data = cropped_data[    t_out_amo : -d_out_amo,  boundary["l_out_amo"] : -r_out_amo, :]
        # elif(len(cropped_data.shape) == 2): cropped_data = cropped_data[    t_out_amo : -d_out_amo,  boundary["l_out_amo"] : -r_out_amo]

        ##### 1. 要把 往外pad  的部分切掉 ( out_amount > 0， 代表往外pad)
        if(l_out_amo > 0):
            if  (len(cropped_data.shape) == 4): cropped_data = cropped_data[ :,                        :                        ,  l_out_amo :                        , :]
            elif(len(cropped_data.shape) == 3): cropped_data = cropped_data[                           :                        ,  l_out_amo :                        , :]
            elif(len(cropped_data.shape) == 2): cropped_data = cropped_data[                           :                        ,  l_out_amo :                        ]
            r_pad_slice -= l_out_amo  ### 更新一下座標
        if(t_out_amo > 0):
            if  (len(cropped_data.shape) == 4): cropped_data = cropped_data[ :,  t_out_amo :                        ,                        :                        , :]
            elif(len(cropped_data.shape) == 3): cropped_data = cropped_data[     t_out_amo :                        ,                        :                        , :]
            elif(len(cropped_data.shape) == 2): cropped_data = cropped_data[     t_out_amo :                        ,                        :                        ]
            d_pad_slice -= t_out_amo  ### 更新一下座標
        if(r_out_amo > 0):
            if  (len(cropped_data.shape) == 4): cropped_data = cropped_data[ :,                        :                        ,                        : -r_out_amo , :]
            elif(len(cropped_data.shape) == 3): cropped_data = cropped_data[                           :                        ,                        : -r_out_amo , :]
            elif(len(cropped_data.shape) == 2): cropped_data = cropped_data[                           :                        ,                        : -r_out_amo ]
        if(d_out_amo > 0):
            if  (len(cropped_data.shape) == 4): cropped_data = cropped_data[ :,                        : -d_out_amo ,                        :                        , :]
            elif(len(cropped_data.shape) == 3): cropped_data = cropped_data[                           : -d_out_amo ,                        :                        , :]
            elif(len(cropped_data.shape) == 2): cropped_data = cropped_data[                           : -d_out_amo ,                        :                        ]


        ##### 2. 要把 往內crop 的部分補回來 ( out_amount == 0， 代表沒有往外pad， 意思就是 保持原樣 或者 往內crop， 要把往內crop的部分補回來，以下的操作 往內crop 漢 保持不變case 都適用， out_amount > 0， 代表有往外pad， 就不用pad囉， 所以 pad_back 就設0 )
        r_pad_back = (back_w - r_pad_slice) if(r_out_amo == 0) else 0  ### slice - slice 會變回格數， img_最右slice - r_pad_slice 即 右邊要pad 回去的格數
        l_pad_back = (l_pad_slice - 0)      if(l_out_amo == 0) else 0  ### slice - slice 會變回格數， l_pad_slice - img_最左slice 即 左邊要pad 回去的格數
        d_pad_back = (back_h - d_pad_slice) if(d_out_amo == 0) else 0  ### slice - slice 會變回格數， img_最下slice - d_pad_slice 即 下邊要pad 回去的格數
        t_pad_back = (t_pad_slice - 0)      if(t_out_amo == 0) else 0  ### slice - slice 會變回格數， t_pad_slice - img_最上slice 即 上邊要pad 回去的格數

        ### pad 回去
        if  (len(cropped_data.shape) == 4): data = tf.pad(cropped_data, ( (0    ,     0), (t_pad_back, d_pad_back), (l_pad_back, r_pad_back), (    0,     0) ) , 'CONSTANT')
        elif(len(cropped_data.shape) == 3): data = tf.pad(cropped_data, ( (t_pad_back, d_pad_back), (l_pad_back, r_pad_back), (    0,     0) )                 , 'CONSTANT')
        elif(len(cropped_data.shape) == 2): data = tf.pad(cropped_data, ( (t_pad_back, d_pad_back), (l_pad_back, r_pad_back) )                                 , 'CONSTANT')
        return data.numpy()
######################################################################################################################################################################################################
######################################################################################################################################################################################################
class Color_jit():
    def __init__(self, do_ratio=0.5):
        '''
        do_ratio ： 要做 color_jit 的機率， 0 ~ 1 之間的數字
        '''
        self.do_ratio   = do_ratio

    def __call__(self, img, Mask, clip_value_min=0, clip_value_max=1):  ### Mask 是想讓背景保持不變，但目前視覺化完後覺得 好像背景跟著變應該會比較好， 先保留好了反正也只是沒用到不影響
        ### 參考https://lufor129.medium.com/tensorflow-%E5%9C%96%E7%89%87%E6%95%B8%E6%93%9A%E5%A2%9E%E5%BC%B7-%E4%BA%8C-%E5%9C%96%E7%89%87%E8%99%95%E7%90%86%E5%87%BD%E6%95%B8-1083fb176f9#44d2
        import tensorflow as tf
        if(tf.random.uniform(shape=[1], minval=0, maxval=1, dtype=tf.float32)[0] <= self.do_ratio):
            ### 視覺化一下 大概的效果長怎樣， 可以拉slider喔覺得寫得還不錯ˊ口ˋ
            # from Dataset_Analyze.step6_tf_aug_analyze import analyze_tf_aug
            # analyze_tf_aug(img=img, Mask=Mask)

            ### 1. 亮度 -0.5 ~ 0.5
            img = tf.image.random_brightness(img, max_delta=0.40)
            ### 2. 對比 0.5 ~ 3
            img = tf.image.random_contrast  (img, lower=0.5, upper=3)

            ### 3. 色相 -0.45 ~ 0.45
            img = tf.image.random_hue       (img, max_delta=0.45)
            ### 4. 飽和 0 ~ 4
            img = tf.image.random_saturation(img, lower=0, upper=4.0)


            ### 把 超出值域的值用clip去掉， 怎麼樣算超出值域 要從外面傳進來喔～
            img = tf.clip_by_value(img, clip_value_min=clip_value_min, clip_value_max=clip_value_max)
        return img

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
        self.knpy_save      = None
        self.add_loss       = None
        self.bgr2rgb        = None

    def __call__(self, model_obj, phase, index, fname, in_ord, in_pre, gt_ord, gt_pre, rec_hope=None, exp_obj=None, training=None, see_reset_init=True, postprocess=False, npz_save=False, knpy_save=False, add_loss=False, bgr2rgb=True):
        self.model_obj      = model_obj
        self.phase          = phase
        self.index          = index
        self.fname          = fname
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
        self.knpy_save      = knpy_save
        self.add_loss       = add_loss
        self.bgr2rgb        = bgr2rgb
        self.doing_things()

    def doing_things(self):
        pass

# class Use_G_generate_Interface:
#     def __call__(self,model_obj, phase, index, in_ord, in_pre, gt_ord, gt_pre, rec_hope=None, exp_obj=None, training=True, see_reset_init=True, postprocess=False, npz_save=False, add_loss=False, bgr2rgb=False):
#         self.doing_things(model_obj, phase, index, in_ord, in_pre, gt_ord, gt_pre, rec_hope=None, exp_obj=None, training=True, see_reset_init=True, postprocess=False, npz_save=False, add_loss=False, bgr2rgb=False)

#     def doing_things(self,model_obj, phase, index, in_ord, in_pre, gt_ord, gt_pre, rec_hope=None, exp_obj=None, training=True, see_reset_init=True, postprocess=False, npz_save=False, add_loss=False, bgr2rgb=False):
#         ''' Not Implement'''
#         pass

######################################################################################################################################################################################################
######################################################################################################################################################################################################

### doc3d v2 [Z: Range(-0.50429183, 0.46694446), Y: Range(-1.2410645, 1.2485291), X: Range(-1.2387834, 1.2280148)]
def wc_save_as_knpy(wc, wc_type, dst_dir, fname, pad_size=20, first_resize=None, final_resize=None, by_the_way_fake_F=False, dis_img=None, dis_img_format=".png", zmin=None, zmax=None, ymin=None, ymax=None, xmin=None, xmax=None, rec_hope=None, rec_hope_format=None):
    '''
    wc_type 有兩種： 
        Wzxy
        Wzyx
    '''
    fname = fname[ :-4]

    dis_img_dir       = f"{dst_dir}/0_dis_img"
    rec_hope_dir      = f"{dst_dir}/0_rec_hope"
    uv_npy_dir        = f"{dst_dir}/1_uv-1_npy"
    uv_knpy_dir       = f"{dst_dir}/1_uv-3_knpy"
    wc_npy_dir        = f"{dst_dir}/2_wc-1_npy"
    W_w_M_npy_dir     = f"{dst_dir}/2_wc-4_W_w_M_npy"
    W_w_M_knpy_dir    = f"{dst_dir}/2_wc-5_W_w_M_knpy"
    W_w_M_visual_dir  = f"{dst_dir}/2_wc-6_W_w_M_visual"
    ppt_visual_dir    = f"{dst_dir}/3_ppt_visual"
    dis_img_path      = f"{dis_img_dir}/{fname}.{dis_img_format}"
    rec_hope_path     = f"{rec_hope_dir}/{fname}.{rec_hope_format}"
    uv_npy_path       = f"{uv_npy_dir}/{fname}.npy"
    uv_knpy_path      = f"{uv_knpy_dir}/{fname}.knpy"
    wc_npy_path       = f"{wc_npy_dir}/{fname}.npy"
    W_w_M_npy_path    = f"{W_w_M_npy_dir}/{fname}.npy"
    W_w_M_knpy_path   = f"{W_w_M_knpy_dir}/{fname}.knpy"
    W_w_M_visual_path = f"{W_w_M_visual_dir}/{fname}.png"
    ppt_visual_path   = f"{ppt_visual_dir}/{fname}.png"
    Check_dir_exist_and_build(wc_npy_dir)
    Check_dir_exist_and_build(W_w_M_npy_dir)
    Check_dir_exist_and_build(W_w_M_knpy_dir)
    Check_dir_exist_and_build(W_w_M_visual_dir)
    Check_dir_exist_and_build(ppt_visual_dir)

    ### 作處理前 要先 resize 到多大
    if(first_resize is not None): wc = cv2.resize(wc, first_resize)

    ### 紀錄一下原始 shape
    h, w, c = wc.shape

    ### 提取 Wx_post, Wy_post, Wz_post
    if  (wc_type.lower() == "Wzyx".lower() ):
        Wz = wc[..., 0:1]
        Wy = wc[..., 1:2]
        Wx = wc[..., 2:3]
    elif(wc_type.lower() == "Wzxy".lower() ):
        Wz = wc[..., 0:1]
        Wx = wc[..., 1:2]
        Wy = wc[..., 2:3]

    ### wc_post 統一轉成 Wzyx

    wc   = np.concatenate([Wz, Wy, Wx], axis=-1)
    Mask = ((wc[..., 0:1] > 0) & (wc[..., 1:2] > 0) & (wc[..., 2:3] > 0)).astype(np.uint8)
    wc_min = np.array([zmin, ymin, xmin]).reshape(1, 1, 3)
    wc_max = np.array([zmax, ymax, xmax]).reshape(1, 1, 3)

    nrows = 1
    ncols = 3
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    ax[0].imshow(cv2.resize(dis_img, (w, h)))
    ax[1].imshow(wc)
    ax[2].imshow(cv2.resize(rec_hope, (w, h)))
    ax[0].axis("off")
    ax[1].axis("off")
    ax[2].axis("off")
    plt.tight_layout()
    plt.savefig(ppt_visual_path)
    # plt.show()


    ### 在 448 的狀態下 做 padding
    wc_post   = np.pad(wc, ( (pad_size, pad_size), (pad_size, pad_size), (0, 0)), "constant" )
    Mask_post = ((wc_post[..., 0:1] > 0) & (wc_post[..., 1:2] > 0) & (wc_post[..., 2:3] > 0)).astype(np.uint8)
    ### 做完處理後 最後 要 resize 到多大
    if(final_resize is not None):
        wc_post   = cv2.resize(wc_post  , final_resize)
        Mask_post = cv2.resize(Mask_post, final_resize)
        Mask_post = Mask_post[..., np.newaxis]  ### 小心 cv2.resize 會自動把ch1去除掉， 所以要記得補回來喔
    ### Mask_post
    # Mask_post = ((wc_post[..., 0:1] > 0) & (wc_post[..., 1:2] > 0) & (wc_post[..., 2:3] > 0)).astype(np.uint8)
    ### 值放回原始range， 記得 * M
    wc_post = ( wc_post * (wc_max - wc_min) + wc_min ) * Mask_post
    W_w_M = np.concatenate( (wc_post, Mask_post), axis= -1 )

    ### 轉 type
    wc_post = wc_post.astype(np.float32)
    W_w_M   = W_w_M  .astype(np.float32)

    ### 存起來
    np.save(wc_npy_path    , wc_post)
    np.save(W_w_M_npy_path , W_w_M)
    Save_npy_path_as_knpy(W_w_M_npy_path, W_w_M_knpy_path)

    ### 視覺化 和 存起來
    ### 每個 ch 取出來 等等視覺化比較好寫
    Wz_post = wc_post[..., 0:1]
    Wy_post = wc_post[..., 1:2]
    Wx_post = wc_post[..., 2:3]


    nrows = 1
    ncols = 5
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))

    ax[0].imshow(wc_post)
    ax[1].imshow(Wz_post)
    ax[2].imshow(Wy_post)
    ax[3].imshow(Wx_post)
    ax[4].imshow(Mask_post)
    plt.tight_layout()

    plt.savefig(W_w_M_visual_path)
    # plt.show()
    plt.close()

    if(by_the_way_fake_F):
        Check_dir_exist_and_build(uv_npy_dir)
        Check_dir_exist_and_build(uv_knpy_dir)

        fake_C = np.zeros(shape=(h, w, 2), dtype=np.float32)
        fake_F = np.concatenate([Mask_post, fake_C], axis=-1)
        np.save(uv_npy_path, fake_F)
        Save_npy_path_as_knpy(uv_npy_path, uv_knpy_path)

    if(dis_img is not None):
        ### 作處理前 要先 resize 到多大
        if(first_resize is not None): dis_img = cv2.resize(dis_img, first_resize)
        ### 在 448 的狀態下 做 padding
        dis_img = np.pad(dis_img, ( (pad_size, pad_size), (pad_size, pad_size), (0,0)), "edge" )
        ### 做完處理後 最後 要 resize 到多大
        if(final_resize is not None): dis_img = cv2.resize(dis_img, final_resize)
        Check_dir_exist_and_build(dis_img_dir)
        cv2.imwrite(dis_img_path, dis_img)
