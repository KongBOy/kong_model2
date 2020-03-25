from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, ReLU, Conv2DTranspose, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from step8_kong_model5_Rect2 import InstanceNorm_kong
# from tensorflow_addons.layers import InstanceNormalization
tf.keras.backend.set_floatx('float32') ### 這步非常非常重要！用了才可以加速！


class MRFBlock(tf.keras.layers.Layer):
    def __init__(self, c_num):
        super(MRFBlock, self).__init__()
        self.conv_11 = Conv2D( c_num, kernel_size=1, strides=1, padding="same")
        self.in_c11  = InstanceNorm_kong()
        self.conv_12 = Conv2D( c_num, kernel_size=1, strides=1, padding="same")
        self.in_c12  = InstanceNorm_kong()

        self.conv_31 = Conv2D( c_num, kernel_size=3, strides=1, padding="same")
        self.in_c31  = InstanceNorm_kong()
        self.conv_32 = Conv2D( c_num, kernel_size=3, strides=1, padding="same")
        self.in_c32  = InstanceNorm_kong()

        self.conv_51 = Conv2D( c_num, kernel_size=5, strides=1, padding="same")
        self.in_c51  = InstanceNorm_kong()
        self.conv_52 = Conv2D( c_num, kernel_size=5, strides=1, padding="same")
        self.in_c52  = InstanceNorm_kong()

        self.conv_71 = Conv2D( c_num, kernel_size=7, strides=1, padding="same")
        self.in_c71  = InstanceNorm_kong()
        self.conv_72 = Conv2D( c_num, kernel_size=7, strides=1, padding="same")
        self.in_c72  = InstanceNorm_kong()

        self.conv_91 = Conv2D( c_num, kernel_size=9, strides=1, padding="same")
        self.in_c91  = InstanceNorm_kong()
        self.conv_92 = Conv2D( c_num, kernel_size=9, strides=1, padding="same")
        self.in_c92  = InstanceNorm_kong()

        self.concat = Concatenate()
    def call(self, input_tensor):
        x1 = self.conv_11(input_tensor)
        x1 = self.in_c11(x1)
        x1 = tf.nn.relu(x1)
        x1 = self.conv_12(x1)
        x1 = self.in_c12(x1)
        x1 = tf.nn.relu(x1)

        x3 = self.conv_31(input_tensor)
        x3 = self.in_c31(x3)
        x3 = tf.nn.relu(x3)
        x3 = self.conv_32(x3)
        x3 = self.in_c32(x3)
        x3 = tf.nn.relu(x3)
        
        x5 = self.conv_51(input_tensor)
        x5 = self.in_c51(x5)
        x5 = tf.nn.relu(x5)
        x5 = self.conv_52(x5)
        x5 = self.in_c52(x5)
        x5 = tf.nn.relu(x5)

        x7 = self.conv_71(input_tensor)
        x7 = self.in_c71(x7)
        x7 = tf.nn.relu(x7)
        x7 = self.conv_72(x7)
        x7 = self.in_c72(x7)
        x7 = tf.nn.relu(x7)

        x9 = self.conv_91(input_tensor)
        x9 = self.in_c91(x9)
        x9 = tf.nn.relu(x9)
        x9 = self.conv_92(x9)
        x9 = self.in_c92(x9)
        x9 = tf.nn.relu(x9)

        x_concat = self.concat([x1, x3, x5, x7, x9])
        return x_concat


if(__name__ == "__main__"):
    import numpy as np
    import matplotlib.pyplot as plt
    # generator = Generator()
    # img_g = np.ones( shape=(1,256,256,3), dtype=np.float32)
    # out_g = generator(img_g)
    # plt.imshow(out_g[0,...])
    # plt.show()
    # print("out_g.numpy()",out_g.numpy())

    # discriminator = Discriminator()
    # img_d1 = np.ones(shape=(1,256,256,3),dtype=np.float32)
    # img_d2 = np.ones(shape=(1,256,256,3),dtype=np.float32)
    # out_d = discriminator(img_d1, img_d2)
    # plt.imshow(out_d[0,...,-1], vmin=-20, vmax=20, cmap='RdBu_r')
    # plt.colorbar()
    # plt.show()
    # print("out_d.numpy()",out_d.numpy())

    mrfb = MRFBlock(c_num=64)
    dis_img = np.ones( shape=(1,384,256,3), dtype=np.float32)
    gt_img  = np.ones( shape=(1,256,256,3), dtype=np.float32)
    optimizer_G = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    optimizer_D = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    summary_writer = tf.summary.create_file_writer( "temp_logs_dir" ) ### 建tensorboard，這會自動建資料夾喔！
    # train_step(rect2, dis_img, gt_img, optimizer_G, optimizer_D, summary_writer, 0)
    mrfb_result = mrfb(dis_img)
    print(mrfb_result)
    print("finish")