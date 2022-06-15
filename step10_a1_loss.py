import tensorflow as tf

import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys
sys.path.append("kong_util")
from kong_util.Disc_and_receptive_field_util import tf_M_resize_then_erosion_by_kong

def norm_to_0_1_by_max_min(data, Mask=None):  ### data 為 np.array才行
    if(Mask is None): return  (data - data.min()) / (data.max() - data.min())
    else:             return ((data - data.min()) / (data.max() - data.min())) * Mask

def mse_kong(gt_data, pred_data, lamb=tf.constant(1., tf.float32), Mask=None):
    n, h, w, c = pred_data.shape     ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
    gt_data = gt_data[:, :h, :w, :]  ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
    if(Mask is None): loss = tf.reduce_mean(tf.math.square(gt_data - pred_data))
    else:
        Mask = Mask[:, :h, :w, :]    ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
        # loss = tf.reduce_sum(tf.math.square((gt_data - pred_data) * Mask)) / ( tf.reduce_sum(Mask) * c)  ### v1 只考慮 batch_size = 1 的狀況， sum 的時機亂寫也沒關係
        # print("loss v1", loss)
        loss = tf.reduce_sum(tf.math.square((gt_data - pred_data) * Mask) / ( tf.reduce_sum(Mask, axis=[1, 2], keepdims=True) * c) ) / n  ### v2 考慮 batch_size = 1 或 >1 的狀況 都可以正確跑囉！
        # print("loss v2", loss)
    return loss * lamb

def mae_kong(gt_data, pred_data, lamb=tf.constant(1., tf.float32), Mask=None):
    n, h, w, c = pred_data.shape     ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
    gt_data = gt_data[:, :h, :w, :]  ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
    if(Mask is None): loss = tf.reduce_mean(tf.math.abs(gt_data - pred_data))
    else:
        Mask = Mask[:, :h, :w, :]    ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
        # loss = tf.reduce_sum(tf.math.abs((gt_data - pred_data) * Mask)) / ( tf.reduce_sum(Mask) * c)
        # print("loss v1", loss)
        loss = tf.reduce_sum(tf.math.abs((gt_data - pred_data) * Mask) / ( tf.reduce_sum(Mask, axis=[1, 2], keepdims=True) * c) ) / n
        # print("loss v2", loss)
    return loss * lamb

class MSE(tf.keras.losses.Loss):
    def __init__(self, mse_scale=1, **args):
        super().__init__(name="MSE")
        self.mse_scale = mse_scale

    def __call__(self, gt_data, pred_data, Mask=None):
        return mse_kong(gt_data, pred_data, self.mse_scale, Mask=Mask)

class MAE(tf.keras.losses.Loss):
    def __init__(self, mae_scale=1, **args):
        super().__init__(name="MAE")
        self.mae_scale = mae_scale

    def __call__(self, gt_data, pred_data, Mask=None):
        print("MAE.__call__.mae_scale:", self.mae_scale)
        return mae_kong(gt_data, pred_data, self.mae_scale, Mask=Mask)

class BCE():
    def __init__(self, bce_scale=1, **args):
        self.bce_scale = bce_scale
        self.tf_fun = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def __call__(self, gt_data, pred_data, Mask=None, Mask_type="Area"):
        print("BCE.__call__.bce_scal:", self.bce_scale)
        n, h, w, c = pred_data.shape     ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
        gt_data = gt_data[:, :h, :w, :]  ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
        bce_loss = -1 * gt_data * tf.math.log(pred_data + 0.0000001) - ( 1 - gt_data) * tf.math.log( 1 - pred_data + 0.0000001)  ### 學tf 加一個小小值防止 log 0 的狀況產生～～嘗試到了　0.0000001 會跟tf2算的一樣
        if(bce_loss.shape[1] != 1 or bce_loss.shape[2] != 1):  ### 如果 hw > 1 的話 要做平均
            if  (Mask is     None): bce_loss = tf.reduce_mean(bce_loss)  ### 無 Mask 的話直接平均
            elif(Mask is not None):  ### 有 Mask 的話， 先把 Mask 縮到相對應的大小， 再根據 Mask 做平均
                n, h, w, c = bce_loss.shape

                ##############################################################################################################################
                ##############################################################################################################################
                ### 嘗試 resize
                # import matplotlib.pyplot as plt

                # bilinear      =  tf.image.resize(Mask, (h, w), method=tf.image.ResizeMethod.BILINEAR)
                # nearest       =  tf.image.resize(Mask, (h, w), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                # bicubic       =  tf.image.resize(Mask, (h, w), method=tf.image.ResizeMethod.BICUBIC)
                # area          =  tf.image.resize(Mask, (h, w), method=tf.image.ResizeMethod.AREA)
                # lanczos3      =  tf.image.resize(Mask, (h, w), method=tf.image.ResizeMethod.LANCZOS3)
                # lanczos5      =  tf.image.resize(Mask, (h, w), method=tf.image.ResizeMethod.LANCZOS5)
                # gaussian      =  tf.image.resize(Mask, (h, w), method=tf.image.ResizeMethod.GAUSSIAN)
                # mitchellcubic =  tf.image.resize(Mask, (h, w), method=tf.image.ResizeMethod.MITCHELLCUBIC)

                # fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
                # ax[0, 0].imshow(bilinear     [0], vmin=0, vmax=1)
                # ax[0, 1].imshow(nearest      [0], vmin=0, vmax=1)
                # ax[0, 2].imshow(bicubic      [0], vmin=0, vmax=1)
                # ax[0, 3].imshow(area         [0], vmin=0, vmax=1)
                # ax[1, 0].imshow(lanczos3     [0], vmin=0, vmax=1)
                # ax[1, 1].imshow(lanczos5     [0], vmin=0, vmax=1)
                # ax[1, 2].imshow(gaussian     [0], vmin=0, vmax=1)
                # ax[1, 3].imshow(mitchellcubic[0], vmin=0, vmax=1)
                # fig.tight_layout()
                ####################################################################################
                ### 嘗試 resize 後再 Erosion的效果
                ### 全1
                # kernel = tf.ones((3, 3, 1))
                ### 自己亂試
                # kernel = tf.constant( [ [[ 0 ], [ 0 ], [ 0 ]],
                #                         [[ 0 ], [ 1 ], [ 0 ]],
                #                         [[ 0 ], [ 0 ], [ 0 ]] ], dtype=tf.float32)
                ### 自己亂試
                # kernel = tf.constant( [ [[0.5], [0.5], [0.5]],
                #                         [[0.5], [ 1 ], [0.5]],
                #                         [[0.5], [0.5], [0.5]] ], dtype=tf.float32)
                ### 高斯kernel
                # kernel = tf.constant( [ [[0.71653131], [0.84648172], [0.71653131]],
                #                         [[0.84648172], [1.        ], [0.84648172]],
                #                         [[0.71653131], [0.84648172], [0.71653131]] ], dtype=tf.float32)
                # bilinear_erosion      =  tf.nn.erosion2d (bilinear     , filters=kernel, strides=(1, 1, 1, 1), padding="SAME", data_format="NHWC", dilations=(1, 1, 1, 1)).numpy()
                # nearest_erosion       =  tf.nn.erosion2d (nearest      , filters=kernel, strides=(1, 1, 1, 1), padding="SAME", data_format="NHWC", dilations=(1, 1, 1, 1)).numpy()
                # bicubic_erosion       =  tf.nn.erosion2d (bicubic      , filters=kernel, strides=(1, 1, 1, 1), padding="SAME", data_format="NHWC", dilations=(1, 1, 1, 1)).numpy()
                # area_erosion          =  tf.nn.erosion2d (area         , filters=kernel, strides=(1, 1, 1, 1), padding="SAME", data_format="NHWC", dilations=(1, 1, 1, 1)).numpy()
                # lanczos3_erosion      =  tf.nn.erosion2d (lanczos3     , filters=kernel, strides=(1, 1, 1, 1), padding="SAME", data_format="NHWC", dilations=(1, 1, 1, 1)).numpy()
                # lanczos5_erosion      =  tf.nn.erosion2d (lanczos5     , filters=kernel, strides=(1, 1, 1, 1), padding="SAME", data_format="NHWC", dilations=(1, 1, 1, 1)).numpy()
                # gaussian_erosion      =  tf.nn.erosion2d (gaussian     , filters=kernel, strides=(1, 1, 1, 1), padding="SAME", data_format="NHWC", dilations=(1, 1, 1, 1)).numpy()
                # mitchellcubic_erosion =  tf.nn.erosion2d (mitchellcubic, filters=kernel, strides=(1, 1, 1, 1), padding="SAME", data_format="NHWC", dilations=(1, 1, 1, 1)).numpy()

                # fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
                # ax[0, 0].imshow(bilinear_erosion     [0] + 1, vmin=0, vmax=1)
                # ax[0, 1].imshow(nearest_erosion      [0] + 1, vmin=0, vmax=1)
                # ax[0, 2].imshow(bicubic_erosion      [0] + 1, vmin=0, vmax=1)
                # ax[0, 3].imshow(area_erosion         [0] + 1, vmin=0, vmax=1)
                # ax[1, 0].imshow(lanczos3_erosion     [0] + 1, vmin=0, vmax=1)
                # ax[1, 1].imshow(lanczos5_erosion     [0] + 1, vmin=0, vmax=1)
                # ax[1, 2].imshow(gaussian_erosion     [0] + 1, vmin=0, vmax=1)
                # ax[1, 3].imshow(mitchellcubic_erosion[0] + 1, vmin=0, vmax=1)
                # fig.tight_layout()
                # print("bilinear_erosion.max()", bilinear_erosion.max())
                # print("bilinear_erosion.min()", bilinear_erosion.min())
                # plt.show()
                ##############################################################################################################################
                ##############################################################################################################################

                if  (Mask_type.lower() == "area")   : Mask = tf.image.resize(Mask, (h, w), method=tf.image.ResizeMethod.AREA)
                elif(Mask_type.lower() == "bicubic"): Mask = tf.image.resize(Mask, (h, w), method=tf.image.ResizeMethod.BICUBIC)
                elif(Mask_type.lower() == "nearest"): Mask = tf.image.resize(Mask, (h, w), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                elif(Mask_type.lower() == "erosion"):
                    Mask = tf_M_resize_then_erosion_by_kong(Mask, resize_h=h, resize_w=w)
                    # kernel = tf.ones((3, 3, 1))
                    # Mask = tf.image.resize(Mask, (h, w), method=tf.image.ResizeMethod.BILINEAR)
                    # if(Mask.shape[1] == 3 and Mask.shape[2] == 3 ):
                    #     Mask = Mask * tf.constant( [[[ 0 ], [ 0 ], [ 0 ]],
                    #                                 [[ 0 ], [ 1 ], [ 0 ]],
                    #                                 [[ 0 ], [ 0 ], [ 0 ]]], dtype=tf.float32)
                    # else:
                    #     Mask = tf.nn.erosion2d(Mask, filters=kernel, strides=(1, 1, 1, 1), padding="SAME", data_format="NHWC", dilations=(1, 1, 1, 1)) + 1
                ##############################################################################################################################
                ##############################################################################################################################
                # import matplotlib.pyplot as plt
                # fig, ax = plt.subplots(nrows=1, ncols=3)
                # ax[0].imshow( bce_loss        [0],         vmin=0, vmax=1)
                # ax[1].imshow( Mask            [0],         vmin=0, vmax=1)
                # ax[2].imshow((bce_loss * Mask)[0], vmin=0, vmax=1)
                # plt.show()
                ##############################################################################################################################
                ##############################################################################################################################

                # bce_loss = tf.reduce_sum( bce_loss * Mask ) / tf.reduce_sum(Mask * c)
                bce_loss = tf.reduce_sum( bce_loss * Mask / ( tf.reduce_sum(Mask, axis=[1, 2], keepdims=True) * c ) ) / n

        # tf_bce_loss = self.tf_fun(gt_data, pred_data)
        # print("   bce_loss",    bce_loss)  ### 確認過和 tf2 一樣
        # print("tf_bce_loss", tf_bce_loss)  ### 確認過和 自己算的一樣
        return bce_loss * self.bce_scale


class Sobel_MAE():
    def __init__(self, sobel_kernel_size, sobel_kernel_scale=1, stride=1, erose_M=False, erose_More=False, **args):
        # super().__init__(name="Sobel_MAE")
        self.sobel_kernel_size  = sobel_kernel_size
        self.sobel_kernel_scale = sobel_kernel_scale
        self.stride       = stride
        self.erose_M      = erose_M
        self.erose_More   = erose_More

    def Visualize_sobel_result(self, sobel_result, vmin=None, vmax=None, Mask=None):
        def _draw_util(fig_cur, ax_cur, data, vmin, vmax):
            ### 畫圖
            ax_img = ax_cur.imshow(data, cmap="gray", vmin=vmin, vmax=vmax)  ### 畫出 數值
            ax_cur.set_title("value: %.4f ~ %.4f" % (vmin, vmax) )           ### 標題標上 min~max

            ### 畫出 colorbar
            divider = make_axes_locatable(ax_cur)  ### 參考：https://matplotlib.org/stable/gallery/axes_grid1/simple_colorbar.html#sphx-glr-gallery-axes-grid1-simple-colorbar-py
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig_cur.colorbar(ax_img, cax=cax, orientation="vertical")


        '''
        輸入的 sobel_result 要 BHWC 喔！
        '''
        if( type(sobel_result)) != type(np.array([])): sobel_result = sobel_result.numpy()

        sobel_result = sobel_result[0]  ### BHWC，所以要 [0]
        h, w, c = sobel_result.shape
        ### 算出 M
        if(Mask is None):
            if   (c == 1): Mask =  (sobel_result[..., 0:1] != 0).astype(np.uint8)
            elif (c == 2): Mask = ((sobel_result[..., 0:1] != 0) & (sobel_result[..., 1:2] != 0)).astype(np.uint8)
            elif (c == 3): Mask = ((sobel_result[..., 0:1] != 0) & (sobel_result[..., 1:2] != 0) & (sobel_result[..., 2:3] != 0)).astype(np.uint8)

        ### (只看M內的部分)
        sobel_result *= Mask

        ### 取出 channel
        if (c >= 1): ch1 = sobel_result[..., 0:1]
        if (c >= 2): ch2 = sobel_result[..., 1:2]
        if (c >= 3): ch3 = sobel_result[..., 2:3]

        ##### 要怎麼視覺化可以自己決定
        ### 最原始的值
        if (c >= 1): ch1 = ch1
        if (c >= 2): ch2 = ch2
        if (c >= 3): ch3 = ch3
        ### 只看值， 不管方向， 直接絕對值 把 正負拿掉
        # if (c >= 1): ch1 = abs(ch1)
        # if (c >= 2): ch2 = abs(ch2)
        # if (c >= 3): ch3 = abs(ch3)
        ### 弄到 01 之間
        # if (c >= 1): ch1 = norm_to_0_1_by_max_min(ch1)
        # if (c >= 2): ch2 = norm_to_0_1_by_max_min(ch2)
        # if (c >= 3): ch3 = norm_to_0_1_by_max_min(ch3)

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        ### matplot 畫出來
        show_size = 5
        nrows = 1
        ncols = c
        if(vmin is None):
            if (c >= 1): vmin_ch1 = ch1.min()
            if (c >= 2): vmin_ch2 = ch2.min()
            if (c >= 3): vmin_ch3 = ch3.min()
        else:
            if (c >= 1): vmin_ch1 = vmin
            if (c >= 2): vmin_ch2 = vmin
            if (c >= 3): vmin_ch3 = vmin
        if(vmax is None):
            if (c >= 1): vmax_ch1 = ch1.max()
            if (c >= 2): vmax_ch2 = ch2.max()
            if (c >= 3): vmax_ch3 = ch3.max()
        else:
            if (c >= 1): vmax_ch1 = vmax
            if (c >= 2): vmax_ch2 = vmax
            if (c >= 3): vmax_ch3 = vmax

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(show_size * ncols, show_size * nrows))
        if  (c == 1):    _draw_util(fig_cur=fig, ax_cur=ax, data=ch1, vmin=vmin_ch1, vmax=vmax_ch1)
        elif( c > 1):
            if (c >= 0): _draw_util(fig_cur=fig, ax_cur=ax[0], data=ch1, vmin=vmin_ch1, vmax=vmax_ch1)
            if (c >= 2): _draw_util(fig_cur=fig, ax_cur=ax[1], data=ch2, vmin=vmin_ch2, vmax=vmax_ch2)
            if (c >= 3): _draw_util(fig_cur=fig, ax_cur=ax[2], data=ch3, vmin=vmin_ch3, vmax=vmax_ch3)

        plt.tight_layout()

        # plt.show()  ### debug用
        return fig, ax

    def _create_sobel_kernel_xy(self):
        # print("doing _create_sobel_kernel_xy")
        # if(self.sobel_kernel_size == 3):
        #     kernels_xy = [ [[-1, -2, -1],
        #                  [ 0,  0,  0],
        #                  [ 1,  2,  1]],   ### matrix_y, (1, 3, 3)
        #                 [[-1,  0,  1],
        #                  [-2,  0,  2],
        #                  [-1,  0,  1]] ]  ### matrix_x, (1, 3, 3)
        # elif(self.sobel_kernel_size == 5):
        #     kernels_xy = [ [[-0.25, -0.4, -0.5,  -0.4, -0.25],
        #                  [-0.20, -0.5,  -1,   -0.5, -0.20],
        #                  [   0,    0,    0,    0,    0   ],
        #                  [ 0.20,  0.5,   1,    0.5,  0.20],
        #                  [ 0.25,  0.4,  0.5,   0.4,  0.25]],   ### matrix_y, (1, 5, 5)
        #                 [[-0.25, -0.2,  0,  0.2, 0.25],
        #                  [-0.40, -0.5,  0,  0.5, 0.40],
        #                  [-0.50,  -1 ,  0,   1 , 0.50],
        #                  [-0.40, -0.5,  0,  0.5, 0.40],
        #                  [-0.25, -0.2,  0,  0.2, 0.25]] ]  ### matrix_x, (1, 5, 5)
        # kernels_xy = np.asarray(kernels_xy, dtype=np.float32)  ### (2, 3, 3)
        # print("kernels_xy", kernels_xy)
        '''
        超棒參考：https://stackoverflow.com/questions/9567882/sobel-filter-kernel-of-large-size

        x方向的梯度， 意思是找出左右變化多的地方， 所以會找出垂直的東西， kernel 看起來也會是垂直的，以 sobel_kernel_size=3 為例
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]]
        y方向的梯度， 意思是找出上下變化多的地方， 所以會找出水平的東西， kernel 看起來也會是水平的，以 sobel_kernel_size=3 為例
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]]
        '''
        center_dist_max = (self.sobel_kernel_size - 1) / 2  ### 以1維來看，離中心最遠的距離， 比如 sobel_kernel_size=5, center_dist_max = 2
        x_1d = np.arange(-center_dist_max, center_dist_max + 1)  ### 比如 sobel_kernel_size=5, x_dir = [-2, -1, 0, 1, 2]
        x_2d = np.expand_dims(x_1d, 0)               ### shape 從 (5) 變 (1, 5)
        x_2d = np.tile(x_2d, (self.sobel_kernel_size, 1))  ### 結果的 shape 為 (5, 5)
        y_2d = x_2d.copy().T

        xy_dist_2d = x_2d ** 2 + y_2d ** 2  ### 算一下 各個點 離中心點的距離 平方， 為什麼平方可以看 StackOverflow， 簡單說 是 本身的距離 * 方向帶有的距離， 所以是 距離平方
        kernel_x = x_2d / xy_dist_2d  ### 距離越遠， 梯度值貢獻越小， 所以用除的， 因為 中心點 會 除以0 變nan， 在下面會統一指定中心算出的nan為0
        kernel_y = y_2d / xy_dist_2d  ### 距離越遠， 梯度值貢獻越小， 所以用除的， 因為 中心點 會 除以0 變nan， 在下面會統一指定中心算出的nan為0
        kernels_xy = np.array( [kernel_x, kernel_y] )  ### 我現在就先放 x 再放 y， 代表 前面找垂直， 後面找水平
        kernels_xy[np.isnan(kernels_xy)] = 0  ### 因為 中心點 會 除以0 變nan， 在這邊統一指定為0喔～

        ##########################################################################################
        # ### 手動指定

        # ### prewitt
        # self.sobel_kernel_size = 3
        # kernels_xy = [ [[-1,  0,  1],
        #              [-1,  0,  1],
        #              [-1,  0,  1]],   ### matrix_y, (1, 3, 3)
        #             [[-1, -1, -1],
        #              [ 0,  0,  0],
        #              [ 1,  1,  1]] ]  ### matrix_x, (1, 3, 3)
        # ############################################
        # ### 模仿 total variance
        # self.sobel_kernel_size = 3
        # kernels_xy = [ [[ 0,  0,  0],
        #              [-1,  0,  1],
        #              [ 0,  0,  0]],   ### matrix_y, (1, 3, 3)
        #             [[ 0, -1,  0],
        #              [ 0,  0,  0],
        #              [ 0,  1,  0]] ]  ### matrix_x, (1, 3, 3)
        # ############################################
        # kernels_xy = np.array(kernels_xy, dtype=np.float32)
        # print("kernels_xy", kernels_xy)
        ##########################################################################################

        kernels_xy = kernels_xy * self.sobel_kernel_scale  ### * self.sobel_kernel_scale 後來根據 StackOverflow 的解釋是說 只是為了方便人看 成一個東西變整數， 其實不需要也沒問題～
        kernels_xy = np.transpose(kernels_xy, (1, 2, 0))  ### (3, 3, 2)
        kernels_xy = np.expand_dims(kernels_xy, -2)       ### (3, 3, 1, 2)
        return kernels_xy

    def Calculate_sobel_edges(self, image, stride=1, Mask=None):
        # print("doing Calculate_sobel_edges")
        '''
        image：BHWC
        kernel：(2, k_size, k_size)
        pad_size： (k_size - 1) / 2
        '''
        n, h, w, c = image.shape

        kernels_xy = self._create_sobel_kernel_xy()                               ### 結果為：(ksize, ksize, 1, 2)， 最後的2裡面 第一個是 x_kernel, 第二個是 y_kernel， 補充一下 寫這邊的主要用意是想要 "有用到的時候再建立"， 如果以平常的寫法的話不好， 因為每次用都要建新的， 但是因注意現在因為有用 @tf.function， 只會在一開始建圖的時候建立一次， 之後不會再建立囉！ 所以這邊可以這樣寫覺得～
        kernels_num = kernels_xy.shape[-1]                                        ### 目前共兩個， x_kernel, y_kernel
        kernels_tf  = tf.constant(kernels_xy, dtype=tf.float32)                   ### numpy 轉 tf
        kernels_tf  = tf.tile(kernels_tf, [1, 1, c, 1], name='sobel_filters')     ### 結果為：(ksize, ksize, C, 2)

        pad_size  = int((self.sobel_kernel_size - 1) / 2)
        pad_sizes = [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]]  ### BHWC
        padded = tf.pad(image, pad_sizes, mode='REFLECT')

        strides = [1, stride, stride, 1]  ### BHWC
        sobel_xy_result = tf.nn.depthwise_conv2d(padded, kernels_tf, strides, 'VALID')    ### 假設 c=3 的例子：(1, 448, 448, 6=3(C)*2(dx 和 dy))
        sobel_xy_result = tf.reshape(sobel_xy_result, shape= (n, h, w, c, kernels_num) )  ### 假設 c=3 的例子：(1, 448, 448, 3, 2 -> 分別是 dx 和 dy)

        sobel_x_result = sobel_xy_result[..., 0]  ### x方向的梯度， 意思是找出左右變化多的地方， 所以會找出垂直的東西
        sobel_y_result = sobel_xy_result[..., 1]  ### y方向的梯度， 意思是找出上下變化多的地方， 所以會找出水平的東西

        ### debug用， 視覺化一下 馬上做完 seobel 後的效果
        # self.Visualize_sobel_result(sobel_x_result)
        # self.Visualize_sobel_result(sobel_y_result)

        ### 通常是用在 WC 上面， 用在 dis_img 感覺沒用， 想法是 把邊緣 大大的值蓋過去， 剩下的就會是 中間的紋理囉
        if(self.erose_M and Mask is not None):
            ### 變小多少 就跟 sobel 用的 kernel_size 一樣大小
            erose_kernel_size = self.sobel_kernel_size
            if(self.erose_More): erose_kernel_size += 8  ### 目前用眼睛看覺得+8不錯
            erose_kernel = tf.ones((self.sobel_kernel_size, self.sobel_kernel_size, 1))
            ### debug用， 顯示 erose前的 M
            # plt.figure()
            # plt.imshow(Mask[0])
            ### 這行做事情： 侵蝕M 讓 M變小一點， 變小多少 就跟 sobel 用的 kernel_size 一樣大小
            Mask = tf.nn.erosion2d (Mask, filters=erose_kernel, strides=(1, 1, 1, 1), padding="SAME", data_format="NHWC", dilations=(1, 1, 1, 1)) + 1  ### 別忘記要 + 1
            ### debug用， 顯示 erose後的 M
            # plt.figure()
            # plt.imshow(Mask[0])
            ### 這兩行做事情： sobel 的結果 乘上縮小的 M， 把邊緣 大大的值蓋過去， 剩下的就會是 中間的紋理囉
            sobel_x_result = sobel_x_result * Mask
            sobel_y_result = sobel_y_result * Mask
            ### debug用， 視覺化一下 乘完後的效果
            # self.Visualize_sobel_result(sobel_x_result)
            # self.Visualize_sobel_result(sobel_y_result)

            ### 用 opencv 內建的 sobel 來比較， 幾乎一模一樣， 只差在 opencv 有多乘上一個係數， kernel_size 最大只支援到 31
            # cv2_sobelx  = cv2.Sobel(image[0].numpy(), cv2.CV_64F, 1, 0, ksize=self.sobel_kernel_size) * Mask[0].numpy()
            # cv2_sobely  = cv2.Sobel(image[0].numpy(), cv2.CV_64F, 0, 1, ksize=self.sobel_kernel_size) * Mask[0].numpy()
            # cv2_sobelxy = cv2.Sobel(image[0].numpy(), cv2.CV_64F, 1, 1, ksize=self.sobel_kernel_size) * Mask[0].numpy()
            # # cv2_sobelx  = cv2.blur(cv2_sobelx, (5, 5))  ### 覺得blur完以後好像 也沒有很有效的消除 grad 的 條紋
            # self.Visualize_sobel_result(cv2_sobelx[np.newaxis, ...])
            # self.Visualize_sobel_result(cv2_sobely[np.newaxis, ...])
            # self.Visualize_sobel_result(cv2_sobelxy[np.newaxis, ...])
            # plt.show()

        return sobel_x_result, sobel_y_result

    # @tf.function  ### GPU debug用
    def __call__(self, gt_data, pred_data, Mask=None):
        n, h, w, c = pred_data.shape     ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
        gt_data = gt_data[:, :h, :w, :]  ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
        if(Mask is not None): Mask    = Mask   [:, :h, :w, :]  ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！

        print("Sobel_MAE.__call__.sobel_kernel_scale:", self.sobel_kernel_scale)
        img1_sobel_x, img1_sobel_y = self.Calculate_sobel_edges(image=gt_data,   Mask=Mask)
        img2_sobel_x, img2_sobel_y = self.Calculate_sobel_edges(image=pred_data, Mask=Mask)

        ### debug用
        # plt.show()
        grad_loss = mae_kong(img1_sobel_x, img2_sobel_x, Mask=Mask) + \
                    mae_kong(img1_sobel_y, img2_sobel_y, Mask=Mask)

        # print("img1_sobel_x.shape", img1_sobel_x.numpy().shape)
        # print("img1_sobel_x.max()", img1_sobel_x.numpy().max())
        # print("img1_sobel_x.min()", img1_sobel_x.numpy().min())
        # print("img1_sobel_y.max()", img1_sobel_y.numpy().max())
        # print("img1_sobel_y.min()", img1_sobel_y.numpy().min())
        # print("grad_x residual:", mae_kong(img1_sobel_x, img2_sobel_x))
        # print("grad_y residual:", mae_kong(img1_sobel_y, img2_sobel_y))
        # print("grad_loss total:", grad_loss)
        # cv2.imwrite( "debug_data/temp_img1_sobel_x.jpg", (norm_to_0_1_by_max_min(img1_sobel_x[0].numpy()) * 255).astype(np.uint8))
        # cv2.imwrite( "debug_data/temp_img1_sobel_y.jpg", (norm_to_0_1_by_max_min(img1_sobel_y[0].numpy()) * 255).astype(np.uint8))
        # cv2.imwrite( "debug_data/temp_img2_sobel_x.jpg", (norm_to_0_1_by_max_min(img2_sobel_x[0].numpy()) * 255).astype(np.uint8))
        # cv2.imwrite( "debug_data/temp_img2_sobel_y.jpg", (norm_to_0_1_by_max_min(img2_sobel_y[0].numpy()) * 255).astype(np.uint8))
        # plt.show()
        return grad_loss


class Total_Variance():
    def __init__(self, tv_scale=1, erose_M=False, erose_More=False, **args):
        self.tv_scale   = tv_scale
        self.erose_M    = erose_M
        self.erose_More = erose_More

    def __call__(self, image, Mask=None):
        n, h, w, c = image.get_shape()
        if(Mask is not None): Mask = Mask[:, :h, :w, :]  ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
        left  = image[:, :,  : -1, :]
        right = image[:, :, 1:   , :]
        top   = image[:,  : -1, :, :]
        down  = image[:, 1:   , :, :]

        x_change = left - right
        y_change = top  - down
        if(Mask is not None):
            Mask_x = Mask[:, :,  : -1, :]
            Mask_y = Mask[:,  : -1, :, :]
            ### 看 Mask有沒有需要侵蝕
            if(self.erose_M):
                erose_kernel_size = 3                                              ### 設定 erose_ksize
                if(self.erose_More): erose_kernel_size += 8                        ### 目前用眼睛看覺得+8不錯
                erose_kernel = tf.ones((erose_kernel_size, erose_kernel_size, 1))  ### 設定 erose_kernel
                ### 這行做事情： 侵蝕M 讓 M變小一點， 變小多少
                Mask_x = tf.nn.erosion2d (Mask_x, filters=erose_kernel, strides=(1, 1, 1, 1), padding="SAME", data_format="NHWC", dilations=(1, 1, 1, 1)) + 1  ### 別忘記要 + 1
                Mask_y = tf.nn.erosion2d (Mask_y, filters=erose_kernel, strides=(1, 1, 1, 1), padding="SAME", data_format="NHWC", dilations=(1, 1, 1, 1)) + 1  ### 別忘記要 + 1

            ### debug用， 看一下 xy_change 長怎樣
            # canvas_size = 5
            # nrows = 1
            # ncols = 6
            # fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(canvas_size * ncols, canvas_size * nrows))
            # ax[0].imshow(x_change[0])
            # ax[1].imshow(y_change[0])
            # ax[2].imshow(Mask[0])
            x_change = x_change * Mask_x
            y_change = y_change * Mask_y
            # ax[3].imshow(x_change[0])
            # ax[4].imshow(y_change[0])
            # ax[5].imshow(Mask_x[0])
            # fig.tight_layout()
            # plt.show()

        x_variance = tf.reduce_mean( tf.abs(x_change) )
        y_variance = tf.reduce_mean( tf.abs(y_change) )

        tv_loss = x_variance + y_variance
        print("Total_Variance.__call__.tv_scale:", self.tv_scale)
        # import matplotlib.pyplot as plt
        # show_size = 5
        # nrows = 2
        # ncols = 2
        # fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(show_size * ncols, show_size * nrows))
        # ax[0, 0].imshow(norm_to_0_1_by_max_min(x_change[0].numpy()))  ### BHWC，所以要 [0]
        # ax[1, 0].imshow(norm_to_0_1_by_max_min(y_change[0].numpy()))  ### BHWC，所以要 [0]
        # plt.tight_layout()
        # plt.show()
        return tv_loss * self.tv_scale

if __name__ == '__main__':
    # window_size = 5
    # pad_size = int((window_size - 1) / 2)
    from step0_access_path import Data_Access_Dir

    '''
    # img1_path = Data_Access_Dir + "2-0b-gt_a_mask.bmp"
    # img2_path = Data_Access_Dir + "2-epoch_0060_a_mask.bmp"
    # img1 = cv2.imread(img1_path)  ### HWC
    # img2 = cv2.imread(img2_path)  ### HWC
    img1_path = "debug_data/" + "1_1_1-pr_Page_141-PZU0001.exr"  ### "1_1_2-cp_Page_0654-XKI0001.exr"
    img2_path = "debug_data/" + "1_1_8-pp_Page_465-YHc0001.exr"  ### "1_1_1-tc_Page_065-YGB0001.exr"
    img1 = cv2.imread(img1_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  ### HWC
    img2 = cv2.imread(img2_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  ### HWC
    # sobel_mae = Sobel_MAE(sobel_kernel_size=3)
    sobel_mae = Sobel_MAE(sobel_kernel_size=5)
    total_variance_loss = Total_Variance()
    ############################################################################################################
    img1 = img1.astype(np.float32)  ### 轉float32
    img2 = img2.astype(np.float32)  ### 轉float32
    img1 = tf.expand_dims(img1, 0)  ### BHWC 這是 丟進 tf_cnn 網路 的標準格式 (1, 448, 448, 3)， 順便直接用 tf.expand_expand_dims 轉成 tensor， 不用 np.expand_dims
    img2 = tf.expand_dims(img2, 0)  ### BHWC 這是 丟進 tf_cnn 網路 的標準格式 (1, 448, 448, 3)， 順便直接用 tf.expand_expand_dims 轉成 tensor， 不用 np.expand_dims
    print(f"img1.shape={img1.shape}")
    print(f"img2.shape={img2.shape}")

    ### 測試用 GradientTape 能不能跑
    with tf.GradientTape() as kong_tape:
        grad_loss_value = sobel_mae(img1, img2)
        tv_loss_value = total_variance_loss(img1)
        grad_loss = grad_loss_value
    print("grad_loss_value:", grad_loss_value)
    print("tv_loss_value:", tv_loss_value)
    generator_gradients = kong_tape .gradient(grad_loss)
    ############################################################################################################
    '''

    W_w_M_v2_1_path = "debug_data/" + "W_w_M_v2-1_2_1-tc_Page_142-OnX0001.npy"
    W_w_M_v2_2_path = "debug_data/" + "W_w_M_v2-1_2_3-pp_Page_171-vRo0001.npy"
    W_w_M_1 = np.load(W_w_M_v2_1_path)
    W_w_M_2 = np.load(W_w_M_v2_2_path)
    W_w_M_1 = W_w_M_1[np.newaxis, ...]
    W_w_M_2 = W_w_M_2[np.newaxis, ...]

    sobel_mae = Sobel_MAE(sobel_kernel_size=3, erose_M=True)
    W_1    = W_w_M_1[..., 0:3]
    W_2    = W_w_M_2[..., 0:3]
    Mask_1 = W_w_M_1[..., 3:4]
    Mask_2 = W_w_M_2[..., 3:4]
    sobel_mae(W_1, W_2, Mask=Mask_1)
    # sobel_mae(W_2, W_1, Mask=Mask_2)
    ### 測試用 GradientTape 能不能跑
    # with tf.GradientTape() as kong_tape:
    #     grad_loss = sobel_mae(W_w_M_1, W_2, Mask=Mask)


    Wz = norm_to_0_1_by_max_min(W_1[0, ..., 0:1])
    Wy = norm_to_0_1_by_max_min(W_1[0, ..., 1:2])
    Wx = norm_to_0_1_by_max_min(W_1[0, ..., 2:3])
    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].imshow(Wz)
    ax[1].imshow(Wx)
    ax[2].imshow(Wy)
    plt.tight_layout()
    plt.show()
