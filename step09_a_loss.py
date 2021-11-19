import tensorflow as tf

import numpy as np
import cv2

def norm_to_0_1_by_max_min(data):  ### data 為 np.array才行
    return (data - data.min()) / (data.max() - data.min())

def mse_kong(tensor1, tensor2, lamb=tf.constant(1., tf.float32)):
    loss = tf.reduce_mean(tf.math.square(tensor1 - tensor2))
    return loss * lamb

def mae_kong(tensor1, tensor2, lamb=tf.constant(1., tf.float32)):
    loss = tf.reduce_mean(tf.math.abs(tensor1 - tensor2))
    return loss * lamb

class MAE(tf.keras.losses.Loss):
    def __init__(self, mae_scale, **args):
        super().__init__(name="MAE")
        self.mae_scale = mae_scale

    def __call__(self, img_true, img_pred):
        return mae_kong(img_true, img_pred, self.mae_scale)

class BCE():
    def __init__(self, bce_scale=1, **args):
        self.bce_scale = bce_scale
        self.tf_fun = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def __call__(self, img_true, img_pred):
        print("self.bce_scale~~~~~~~~~~~~~", self.bce_scale)
        bce_loss = self.tf_fun(img_true, img_pred)
        return bce_loss * self.bce_scale


class Sobel_MAE(tf.keras.losses.Loss):
    def __init__(self, kernel_size, kernel_scale=1, stride=1, **args):
        super().__init__(name="Sobel_MAE")
        self.kernel_size  = kernel_size
        self.kernel_scale = kernel_scale
        self.stride       = stride

    def _create_sobel_kernel_xy(self):
        print("doing _create_sobel_kernel_xy")
        # if(self.kernel_size == 3):
        #     kernels = [ [[-1, -2, -1],
        #                  [ 0,  0,  0],
        #                  [ 1,  2,  1]],   ### matrix_y, (1, 3, 3)
        #                 [[-1,  0,  1],
        #                  [-2,  0,  2],
        #                  [-1,  0,  1]] ]  ### matrix_x, (1, 3, 3)
        # elif(self.kernel_size == 5):
        #     kernels = [ [[-0.25, -0.4, -0.5,  -0.4, -0.25],
        #                  [-0.20, -0.5,  -1,   -0.5, -0.20],
        #                  [   0,    0,    0,    0,    0   ],
        #                  [ 0.20,  0.5,   1,    0.5,  0.20],
        #                  [ 0.25,  0.4,  0.5,   0.4,  0.25]],   ### matrix_y, (1, 5, 5)
        #                 [[-0.25, -0.2,  0,  0.2, 0.25],
        #                  [-0.40, -0.5,  0,  0.5, 0.40],
        #                  [-0.50,  -1 ,  0,   1 , 0.50],
        #                  [-0.40, -0.5,  0,  0.5, 0.40],
        #                  [-0.25, -0.2,  0,  0.2, 0.25]] ]  ### matrix_x, (1, 5, 5)
        # kernels = np.asarray(kernels, dtype=np.float32)  ### (2, 3, 3)
        # print("kernels", kernels)
        '''
        超棒參考：https://stackoverflow.com/questions/9567882/sobel-filter-kernel-of-large-size

        x方向的梯度， 意思是找出左右變化多的地方， 所以會找出垂直的東西， kernel 看起來也會是垂直的，以 kernel_size=3 為例
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]]
        y方向的梯度， 意思是找出上下變化多的地方， 所以會找出水平的東西， kernel 看起來也會是水平的，以 kernel_size=3 為例
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]]
        '''
        center_dist_max = (self.kernel_size - 1) / 2  ### 以1維來看，離中心最遠的距離， 比如 kernel_size=5, center_dist_max = 2
        x_1d = np.arange(-center_dist_max, center_dist_max + 1)  ### 比如 kernel_size=5, x_dir = [-2, -1, 0, 1, 2]
        x_2d = np.expand_dims(x_1d, 0)               ### shape 從 (5) 變 (1, 5)
        x_2d = np.tile(x_2d, (self.kernel_size, 1))  ### 結果的 shape 為 (5, 5)
        y_2d = x_2d.copy().T

        xy_dist_2d = x_2d ** 2 + y_2d ** 2  ### 算一下 各個點 離中心點的距離 平方， 為什麼平方可以看 StackOverflow， 簡單說 是 本身的距離 * 方向帶有的距離， 所以是 距離平方
        kernel_x = x_2d / xy_dist_2d  ### 距離越遠， 梯度值貢獻越小， 所以用除的， 因為 中心點 會 除以0 變nan， 在下面會統一指定中心算出的nan為0
        kernel_y = y_2d / xy_dist_2d  ### 距離越遠， 梯度值貢獻越小， 所以用除的， 因為 中心點 會 除以0 變nan， 在下面會統一指定中心算出的nan為0
        kernels = np.array( [kernel_x, kernel_y] )  ### 我現在就先放 x 再放 y， 代表 前面找垂直， 後面找水平
        kernels[np.isnan(kernels)] = 0  ### 因為 中心點 會 除以0 變nan， 在這邊統一指定為0喔～

        ##########################################################################################
        # ### 手動指定

        # ### prewitt
        # self.kernel_size = 3
        # kernels = [ [[-1,  0,  1],
        #              [-1,  0,  1],
        #              [-1,  0,  1]],   ### matrix_y, (1, 3, 3)
        #             [[-1, -1, -1],
        #              [ 0,  0,  0],
        #              [ 1,  1,  1]] ]  ### matrix_x, (1, 3, 3)
        # ############################################
        # ### 模仿 total variance
        # self.kernel_size = 3
        # kernels = [ [[ 0,  0,  0],
        #              [-1,  0,  1],
        #              [ 0,  0,  0]],   ### matrix_y, (1, 3, 3)
        #             [[ 0, -1,  0],
        #              [ 0,  0,  0],
        #              [ 0,  1,  0]] ]  ### matrix_x, (1, 3, 3)
        # ############################################
        # kernels = np.array(kernels, dtype=np.float32)
        # print("kernels", kernels)
        ##########################################################################################

        kernels = kernels * self.kernel_scale  ### * self.kernel_scale 後來根據 StackOverflow 的解釋是說 只是為了方便人看 成一個東西變整數， 其實不需要也沒問題～
        kernels = np.transpose(kernels, (1, 2, 0))  ### (3, 3, 2)
        kernels = np.expand_dims(kernels, -2)       ### (3, 3, 1, 2)
        return kernels

    def Calculate_sobel_edges(self, image, stride=1):
        print("doing Calculate_sobel_edges")
        '''
        image：BHWC
        kernel：(2, k_size, k_size)
        pad_size： (k_size - 1) / 2
        '''
        image_shape = image.get_shape()

        kernels = self._create_sobel_kernel_xy()  ### 寫這邊的主要用意是想要 "有用到的時候再建立"， 如果以平常的寫法的話不好， 因為每次用都要建新的， 但是因注意現在因為有用 @tf.function， 只會在一開始建圖的時候建立一次， 之後不會再建立囉！ 所以這邊可以這樣寫覺得～
        kernels_num = kernels.shape[-1]
        kernels_tf  = tf.constant(kernels, dtype=tf.float32)
        kernels_tf  = tf.tile(kernels_tf, [1, 1, image_shape[-1], 1], name='sobel_filters')  ### (3, 3, C, 2)

        pad_size  = int((self.kernel_size - 1) / 2)
        pad_sizes = [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]]  ### BHWC
        padded = tf.pad(image, pad_sizes, mode='REFLECT')

        strides = [1, stride, stride, 1]  ### BHWC
        output = tf.nn.depthwise_conv2d(padded, kernels_tf, strides, 'VALID')  ### (1, 448, 448, 6=3(C)*2(dx 和 dy))
        output = tf.reshape(output, shape=image_shape + [kernels_num])  ### (1, 448, 448, 3, 2 -> 分別是 dx 和 dy)
        return output

    def call(self, img1, img2):
        print("doing sobel_mae_loss, kernel_scale=", self.kernel_scale)
        img1_sobel_xy = self.Calculate_sobel_edges(image=img1)
        img1_sobel_x = img1_sobel_xy[..., 0]  ### x方向的梯度， 意思是找出左右變化多的地方， 所以會找出垂直的東西
        img1_sobel_y = img1_sobel_xy[..., 1]  ### y方向的梯度， 意思是找出上下變化多的地方， 所以會找出水平的東西

        img2_sobel_xy = self.Calculate_sobel_edges(image=img2)
        img2_sobel_x = img2_sobel_xy[..., 0]  ### x方向的梯度， 意思是找出左右變化多的地方， 所以會找出垂直的東西
        img2_sobel_y = img2_sobel_xy[..., 1]  ### y方向的梯度， 意思是找出上下變化多的地方， 所以會找出水平的東西
        grad_loss = mae_kong(img1_sobel_x, img2_sobel_x) + mae_kong(img1_sobel_y, img2_sobel_y)

        # import matplotlib.pyplot as plt
        # show_size = 5
        # nrows = 2
        # ncols = 2
        # fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(show_size * ncols, show_size * nrows))
        # ax[0, 0].imshow(norm_to_0_1_by_max_min(img1_sobel_x[0].numpy()))  ### BHWC，所以要 [0]
        # ax[0, 1].imshow(norm_to_0_1_by_max_min(img1_sobel_y[0].numpy()))  ### BHWC，所以要 [0]
        # ax[1, 0].imshow(norm_to_0_1_by_max_min(img2_sobel_x[0].numpy()))  ### BHWC，所以要 [0]
        # ax[1, 1].imshow(norm_to_0_1_by_max_min(img2_sobel_y[0].numpy()))  ### BHWC，所以要 [0]
        # plt.tight_layout()
        # print("img1_sobel_x.max()", img1_sobel_x.numpy().max())
        # print("img1_sobel_x.min()", img1_sobel_x.numpy().min())
        # print("img1_sobel_y.max()", img1_sobel_y.numpy().max())
        # print("img1_sobel_y.min()", img1_sobel_y.numpy().min())
        # cv2.imwrite( "temp_img1_sobel_x.jpg", (norm_to_0_1_by_max_min(img1_sobel_x[0].numpy()) * 255).astype(np.uint8))
        # cv2.imwrite( "temp_img1_sobel_y.jpg", (norm_to_0_1_by_max_min(img1_sobel_y[0].numpy()) * 255).astype(np.uint8))
        # cv2.imwrite( "temp_img2_sobel_x.jpg", (norm_to_0_1_by_max_min(img2_sobel_x[0].numpy()) * 255).astype(np.uint8))
        # cv2.imwrite( "temp_img2_sobel_y.jpg", (norm_to_0_1_by_max_min(img2_sobel_y[0].numpy()) * 255).astype(np.uint8))
        # plt.show()
        return grad_loss


class Total_Variance():
    def __init__(self, tv_scale=1, **args):
        self.tv_scale = tv_scale

    def __call__(self, image):
        n, h, w, c = image.get_shape()
        left  = image[:, :,  : -1, :]
        right = image[:, :, 1:   , :]
        top   = image[:,  : -1, :, :]
        down  = image[:, 1:   , :, :]

        x_change = left - right
        y_change = top  - down
        x_variance = tf.reduce_mean( tf.abs(x_change) )
        y_variance = tf.reduce_mean( tf.abs(y_change) )

        tv_loss = x_variance + y_variance
        print("self.tv_scale~~~~~~~~~~~~~", self.tv_scale)
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

    img1_path = "1_1_8-pp_Page_465-YHc0001.exr"  ### "1_1_2-cp_Page_0654-XKI0001.exr"
    img2_path = "1_1_1-pr_Page_141-PZU0001.exr"  ### "1_1_1-tc_Page_065-YGB0001.exr"
    img1 = cv2.imread(img1_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  ### HWC
    img2 = cv2.imread(img2_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  ### HWC

    # img1_path = "2-0b-gt_a_mask.bmp"
    # img2_path = "2-epoch_0060_a_mask.bmp"
    # img1 = cv2.imread(img1_path)  ### HWC
    # img2 = cv2.imread(img2_path)  ### HWC

    ### debug用
    # cv2.imshow("img1", img1)
    # cv2.imshow("img2", img2)


    # sobel_mae = Sobel_MAE(kernel_size=3)
    sobel_mae = Sobel_MAE(kernel_size=5)
    total_variance_loss = Total_Variance()
    ############################################################################################################
    img1 = img1.astype(np.float32)  ### 轉float32
    img2 = img2.astype(np.float32)  ### 轉float32
    img1 = tf.expand_dims(img1, 0)  ### BHWC 這是 丟進 tf_cnn 網路 的標準格式 (1, 448, 448, 3)， 順便直接用 tf.expand_expand_dims 轉成 tensor， 不用 np.expand_dims
    img2 = tf.expand_dims(img2, 0)  ### BHWC 這是 丟進 tf_cnn 網路 的標準格式 (1, 448, 448, 3)， 順便直接用 tf.expand_expand_dims 轉成 tensor， 不用 np.expand_dims
    print(f"img1.shape={img1.shape}")
    print(f"img2.shape={img2.shape}")
    ############################################################################################################

    with tf.GradientTape() as kong_tape:
        grad_loss_value = sobel_mae(img1, img2)
        tv_loss_value = total_variance_loss(img1)
    print("grad_loss_value:", grad_loss_value)
    print("tv_loss_value:", tv_loss_value)
    # generator_gradients = kong_tape .gradient(grad_loss)
