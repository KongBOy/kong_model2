import tensorflow as tf

import numpy as np
import cv2

def mse_kong(tensor1, tensor2, lamb=tf.constant(1., tf.float32)):
    loss = tf.reduce_mean(tf.math.square(tensor1 - tensor2))
    return loss * lamb

def mae_kong(tensor1, tensor2, lamb=tf.constant(1., tf.float32)):
    loss = tf.reduce_mean(tf.math.abs(tensor1 - tensor2))
    return loss * lamb


def sobel_edges(image, kernels, pad_size, stride=1):
    '''
    image：BHWC
    kernel：(2, k_size, k_size)
    pad_size： (k_size - 1) / 2
    '''
    image_shape = image.get_shape()
    kernels_num = len(kernels)
    kernels     = np.asarray(kernels)               ### (2, 3, 3)
    kernels     = np.transpose(kernels, (1, 2, 0))  ### (3, 3, 2)
    kernels     = np.expand_dims(kernels, -2)       ### (3, 3, 1, 2)
    kernels_tf  = tf.constant(kernels, dtype=tf.float32)
    kernels_tf  = tf.tile(kernels_tf, [1, 1, image_shape[-1], 1], name='sobel_filters')  ### (3, 3, C, 2)

    pad_sizes = [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]]  ### BHWC
    padded = tf.pad(image, pad_sizes, mode='REFLECT')

    strides = [1, stride, stride, 1]  ### BHWC
    output = tf.nn.depthwise_conv2d(padded, kernels_tf, strides, 'VALID')  ### (1, 448, 448, 6)
    output = tf.reshape(output, shape=image_shape + [kernels_num])  ### (1, 448, 448, 3, 2->分別是 dx 和 dy)
    return output

def sobel_mae_loss(img1, img2, kernels=[ [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] ], pad_size=1):
    img1_sobel_xy = sobel_edges(image=img1, kernels=kernels, pad_size=pad_size)
    img1_sobel_y = img1_sobel_xy[..., 0]
    img1_sobel_x = img1_sobel_xy[..., 1]

    img2_sobel_xy = sobel_edges(image=img2, kernels=kernels, pad_size=pad_size)
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

if __name__ == '__main__':
    window_size = 3
    pad_size = int((window_size - 1) / 2)

    img1_path = "1_1_8-pp_Page_465-YHc0001.exr"  ### "1_1_2-cp_Page_0654-XKI0001.exr"
    img2_path = "1_1_1-pr_Page_141-PZU0001.exr"  ### "1_1_1-tc_Page_065-YGB0001.exr"
    img1 = cv2.imread(img1_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  ### HWC
    img2 = cv2.imread(img2_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  ### HWC
    kernels = [ [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],   ### matrix_y, (2, 3, 3)
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] ]  ### matrix_x, (2, 3, 3)

    ############################################################################################################
    img1 = tf.expand_dims(img1, 0)  ### BHWC 這是 丟進 tf_cnn 網路 的標準格式 (1, 448, 448, 3)， 順便直接用 tf.expand_expand_dims 轉成 tensor， 不用 np.expand_dims
    img2 = tf.expand_dims(img2, 0)  ### BHWC 這是 丟進 tf_cnn 網路 的標準格式 (1, 448, 448, 3)， 順便直接用 tf.expand_expand_dims 轉成 tensor， 不用 np.expand_dims
    print(f"img1.shape={img1.shape}")
    print(f"img2.shape={img2.shape}")

    with tf.GradientTape() as kong_tape:
        grad_loss_value = sobel_mae_loss(img1, img2, kernels, pad_size)
    print(grad_loss_value)
    # generator_gradients = kong_tape .gradient(grad_loss)