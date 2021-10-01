import tensorflow as tf

import numpy as np
import cv2

def mse_kong(tensor1, tensor2, lamb=tf.constant(1., tf.float32)):
    loss = tf.reduce_mean(tf.math.square(tensor1 - tensor2))
    return loss * lamb

def mae_kong(tensor1, tensor2, lamb=tf.constant(1., tf.float32)):
    loss = tf.reduce_mean(tf.math.abs(tensor1 - tensor2))
    return loss * lamb

class Sobel_MAE(tf.keras.losses.Loss):
    def __init__(self, kernel_size, kernel_scale=1, stride=1):
        super().__init__(name="Sobel_MAE")
        self.kernel_size  = kernel_size
        self.kernel_scale = kernel_scale
        self.stride       = stride

    def create_sobel_kernel(self):
        if(self.kernel_size == 3):
            kernels = [ [[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]],   ### matrix_y, (1, 3, 3)
                        [[-1,  0,  1],
                         [-2,  0,  2],
                         [-1,  0,  1]] ]  ### matrix_x, (1, 3, 3)
        elif(self.kernel_size == 5):
            kernels = [ [[-0.25, -0.4, -0.5,  -0.4, -0.25],
                         [-0.20, -0.5,  -1,   -0.5, -0.20],
                         [   0,    0,    0,    0,    0   ],
                         [ 0.20,  0.5,   1,    0.5,  0.20],
                         [ 0.25,  0.4,  0.5,   0.4,  0.25]],   ### matrix_y, (1, 5, 5)
                        [[-0.25, -0.2,  0,  0.2, 0.25],
                         [-0.40, -0.5,  0,  0.5, 0.40],
                         [-0.50,  -1 ,  0,   1 , 0.50],
                         [-0.40, -0.5,  0,  0.5, 0.40],
                         [-0.25, -0.2,  0,  0.2, 0.25]] ]  ### matrix_x, (1, 5, 5)
        kernels = np.asarray(kernels, dtype=np.float32)  ### (2, 3, 3)
        kernels = kernels * self.kernel_scale
        kernels = np.transpose(kernels, (1, 2, 0))  ### (3, 3, 2)
        kernels = np.expand_dims(kernels, -2)       ### (3, 3, 1, 2)
        return kernels

    def Calculate_sobel_edges(self, image, stride=1):
        print("doing sobel_mae_loss")
        '''
        image：BHWC
        kernel：(2, k_size, k_size)
        pad_size： (k_size - 1) / 2
        '''
        image_shape = image.get_shape()

        kernels = self.create_sobel_kernel()  ### 寫這邊的主要用意是想要 "有用到的時候再建立"， 如果以平常的寫法的話不好， 因為每次用都要建新的， 但是因注意現在因為有用 @tf.function， 只會在一開始建圖的時候建立一次， 之後不會再建立囉！ 所以這邊可以這樣寫覺得～
        kernels_num = kernels.shape[-1]
        kernels_tf  = tf.constant(kernels, dtype=tf.float32)
        kernels_tf  = tf.tile(kernels_tf, [1, 1, image_shape[-1], 1], name='sobel_filters')  ### (3, 3, C, 2)

        pad_size  = int((self.kernel_size - 1) / 2)
        pad_sizes = [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]]  ### BHWC
        padded = tf.pad(image, pad_sizes, mode='REFLECT')

        strides = [1, stride, stride, 1]  ### BHWC
        output = tf.nn.depthwise_conv2d(padded, kernels_tf, strides, 'VALID')  ### (1, 448, 448, 6)
        output = tf.reshape(output, shape=image_shape + [kernels_num])  ### (1, 448, 448, 3, 2 -> 分別是 dx 和 dy)
        return output


    def call(self, img1, img2):
        print("doing sobel_mae_loss")
        img1_sobel_xy = self.Calculate_sobel_edges(image=img1)
        img1_sobel_y = img1_sobel_xy[..., 0]
        img1_sobel_x = img1_sobel_xy[..., 1]

        img2_sobel_xy = self.Calculate_sobel_edges(image=img2)
        img2_sobel_y = img2_sobel_xy[..., 0]
        img2_sobel_x = img2_sobel_xy[..., 1]
        grad_loss = mae_kong(img1_sobel_x, img2_sobel_x) + mae_kong(img1_sobel_y, img2_sobel_y)

        # import matplotlib.pyplot as plt
        # show_size = 5
        # nrows = 2
        # ncols = 2
        # fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(show_size * ncols, show_size * nrows))
        # ax[0, 0].imshow(img1_sobel_y[0])  ### BHWC，所以要 [0]
        # ax[0, 1].imshow(img1_sobel_x[0])  ### BHWC，所以要 [0]
        # ax[1, 0].imshow(img2_sobel_y[0])  ### BHWC，所以要 [0]
        # ax[1, 1].imshow(img2_sobel_x[0])  ### BHWC，所以要 [0]
        # plt.tight_layout()
        # plt.show()
        return grad_loss

# def create_sobel_kernel(kernel_size, kernel_scale):
#     if(kernel_size == 3):
#         kernels = [ [[-1, -2, -1],
#                      [ 0,  0,  0],
#                      [ 1,  2,  1]],   ### matrix_y, (1, 3, 3)
#                     [[-1,  0,  1],
#                      [-2,  0,  2],
#                      [-1,  0,  1]] ]  ### matrix_x, (1, 3, 3)
#     elif(kernel_size == 5):
#         kernels = [ [[0.25, 0.4,  0,  0.4, 0.25],
#                      [0.20, 0.5,  0,  0.5, 0.20],
#                      [  0,   0,   0,    0,    0],
#                      [0.20, 0.5,  1,  0.5, 0.20],
#                      [0.25, 0.4,  0,  0.4, 0.25]],   ### matrix_y, (1, 5, 5)
#                     [[0.25, 0.2,  0,  0.2, 0.25],
#                      [0.40, 0.5,  0,  0.5, 0.40],
#                      [0.50,  1 ,  0,   1 , 0.50],
#                      [0.40, 0.5,  0,  0.5, 0.40],
#                      [0.25, 0.2,  0,  0.2, 0.25]] ]  ### matrix_x, (1, 5, 5)
#     return kernels * kernel_scale

# def sobel_edges(image, kernel_size, kernel_scale=1, stride=1):
#     print("doing sobel_mae_loss")
#     '''
#     image：BHWC
#     kernel：(2, k_size, k_size)
#     pad_size： (k_size - 1) / 2
#     '''
#     image_shape = image.get_shape()

#     kernels = create_sobel_kernel(kernel_size, kernel_scale)
#     kernels_num = len(kernels)
#     kernels     = np.asarray(kernels, dtype=np.float32)               ### (2, 3, 3)
#     kernels     = np.transpose(kernels, (1, 2, 0))  ### (3, 3, 2)
#     kernels     = np.expand_dims(kernels, -2)       ### (3, 3, 1, 2)
#     kernels_tf  = tf.constant(kernels, dtype=tf.float32)
#     kernels_tf  = tf.tile(kernels_tf, [1, 1, image_shape[-1], 1], name='sobel_filters')  ### (3, 3, C, 2)

#     pad_size  = int((kernel_size - 1) / 2)
#     pad_sizes = [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]]  ### BHWC
#     padded = tf.pad(image, pad_sizes, mode='REFLECT')

#     strides = [1, stride, stride, 1]  ### BHWC
#     output = tf.nn.depthwise_conv2d(padded, kernels_tf, strides, 'VALID')  ### (1, 448, 448, 6)
#     output = tf.reshape(output, shape=image_shape + [kernels_num])  ### (1, 448, 448, 3, 2 -> 分別是 dx 和 dy)
#     return output

# def sobel_mae_loss(img1, img2, kernel_size=3):
#     print("doing sobel_mae_loss")
#     img1_sobel_xy = sobel_edges(image=img1, kernel_size=kernel_size)
#     img1_sobel_y = img1_sobel_xy[..., 0]
#     img1_sobel_x = img1_sobel_xy[..., 1]

#     img2_sobel_xy = sobel_edges(image=img2, kernel_size=kernel_size)
#     img2_sobel_y = img2_sobel_xy[..., 0]
#     img2_sobel_x = img2_sobel_xy[..., 1]
#     grad_loss = mae_kong(img1_sobel_x, img2_sobel_x) + mae_kong(img1_sobel_y, img2_sobel_y)

#     import matplotlib.pyplot as plt
#     show_size = 5
#     nrows = 2
#     ncols = 2
#     fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(show_size * ncols, show_size * nrows))
#     ax[0, 0].imshow(img1_sobel_y[0])  ### BHWC，所以要 [0]
#     ax[0, 1].imshow(img1_sobel_x[0])  ### BHWC，所以要 [0]
#     ax[1, 0].imshow(img2_sobel_y[0])  ### BHWC，所以要 [0]
#     ax[1, 1].imshow(img2_sobel_x[0])  ### BHWC，所以要 [0]
#     plt.tight_layout()
#     plt.show()
#     return grad_loss

if __name__ == '__main__':
    window_size = 5
    pad_size = int((window_size - 1) / 2)

    img1_path = "1_1_8-pp_Page_465-YHc0001.exr"  ### "1_1_2-cp_Page_0654-XKI0001.exr"
    img2_path = "1_1_1-pr_Page_141-PZU0001.exr"  ### "1_1_1-tc_Page_065-YGB0001.exr"
    img1 = cv2.imread(img1_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  ### HWC
    img2 = cv2.imread(img2_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  ### HWC

    sobel_k3_mae = Sobel_MAE(kernel_size=5)
    ############################################################################################################
    img1 = tf.expand_dims(img1, 0)  ### BHWC 這是 丟進 tf_cnn 網路 的標準格式 (1, 448, 448, 3)， 順便直接用 tf.expand_expand_dims 轉成 tensor， 不用 np.expand_dims
    img2 = tf.expand_dims(img2, 0)  ### BHWC 這是 丟進 tf_cnn 網路 的標準格式 (1, 448, 448, 3)， 順便直接用 tf.expand_expand_dims 轉成 tensor， 不用 np.expand_dims
    print(f"img1.shape={img1.shape}")
    print(f"img2.shape={img2.shape}")

    with tf.GradientTape() as kong_tape:
        # grad_loss_value = sobel_mae_loss(img1, img2, 3)
        grad_loss_value = sobel_k3_mae(img1, img2)
    print(grad_loss_value)
    # generator_gradients = kong_tape .gradient(grad_loss)
